"""
POD2: Cropping engine for extracting regions from orthophotos
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box, mapping, shape
from shapely.ops import unary_union

from ...core.config import get_config
from ...core.logger import get_logger
from ...core.utils import FileUtils, GeoUtils

logger = get_logger(__name__)


class CroppingEngine:
    """Main engine for POD2 cropping operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize cropping engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_config().get('cropping')
        
        # Setup directories
        self.input_dir = get_config().output_dir / "pod1_output"
        self.output_dir = get_config().output_dir / "pod2_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.use_convex_hull = self.config.get('use_convex_hull', True)
        self.buffer_size = self.config.get('buffer_size', 10)  # meters
        self.min_area = self.config.get('min_area', 100)  # square meters
        self.validate_geometry = self.config.get('validate_geometry', True)
        
        # Initialize results registry
        self.results = {
            'cropped_images': [],
            'cropped_regions': [],
            'statistics': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        logger.info("POD2 Cropping Engine initialized")
    
    def process(self, 
                image_source: Optional[Union[str, Path]] = None,
                geometry_source: Optional[Union[str, Path, gpd.GeoDataFrame]] = None) -> Dict[str, Any]:
        """Main processing method for cropping
        
        Args:
            image_source: Path to image or directory (uses POD1 output if None)
            geometry_source: Path to shapefile, GeoDataFrame, or directory
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting POD2 cropping process")
        
        try:
            # Load POD1 registry if using default input
            if image_source is None:
                registry_path = self.input_dir / "registry.json"
                if registry_path.exists():
                    with open(registry_path, 'r') as f:
                        pod1_registry = json.load(f)
                        image_source = self.input_dir / "images"
                        
                        # Get geometry from POD1 registry
                        if geometry_source is None and pod1_registry.get('shapefiles'):
                            for shp_info in pod1_registry['shapefiles'].values():
                                if 'output_path' in shp_info:
                                    geometry_source = shp_info['output_path']
                                    break
            
            # Get list of images to process
            images = self._get_image_list(image_source)
            logger.info(f"Found {len(images)} images to process")
            
            # Load geometries for cropping
            geometries = self._load_geometries(geometry_source)
            logger.info(f"Loaded {len(geometries)} geometries for cropping")
            
            # Process each image with each geometry
            for img_path in images:
                logger.info(f"Processing image: {img_path}")
                self._crop_image_by_geometries(img_path, geometries)
            
            # Calculate statistics
            self._calculate_statistics()
            
            # Save results
            self._save_results()
            
            self.results['status'] = 'completed'
            self.results['output_path'] = str(self.output_dir)
            
            logger.info(f"POD2 processing completed. Output: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error in POD2 processing: {str(e)}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
        
        return self.results
    
    def _get_image_list(self, source: Union[str, Path]) -> List[Path]:
        """Get list of images to process
        
        Args:
            source: Image source path
            
        Returns:
            List of image paths
        """
        source = Path(source) if source else self.input_dir / "images"
        
        if not source.exists():
            logger.warning(f"Image source does not exist: {source}")
            return []
        
        if source.is_file():
            return [source]
        else:
            # List all TIF files
            images = list(source.glob("*.tif"))
            images.extend(source.glob("*.tiff"))
            return sorted(images)
    
    def _load_geometries(self, source: Union[str, Path, gpd.GeoDataFrame]) -> List[Dict]:
        """Load geometries for cropping
        
        Args:
            source: Geometry source
            
        Returns:
            List of geometry dictionaries
        """
        geometries = []
        
        if isinstance(source, gpd.GeoDataFrame):
            gdf = source
        elif source:
            source = Path(source)
            if source.exists():
                if source.is_file():
                    gdf = gpd.read_file(source)
                else:
                    # Load all shapefiles in directory
                    shp_files = list(source.glob("*.shp"))
                    if shp_files:
                        gdf = gpd.read_file(shp_files[0])
                    else:
                        logger.warning(f"No shapefiles found in: {source}")
                        return geometries
            else:
                logger.warning(f"Geometry source does not exist: {source}")
                return geometries
        else:
            logger.warning("No geometry source provided")
            return geometries
        
        # Process each geometry
        for idx, row in gdf.iterrows():
            try:
                geom = row.geometry
                
                # Skip invalid geometries
                if not geom or not geom.is_valid:
                    if self.validate_geometry:
                        logger.warning(f"Skipping invalid geometry at index {idx}")
                        continue
                    else:
                        # Try to fix invalid geometry
                        geom = geom.buffer(0)
                
                # Skip small geometries
                if geom.area < self.min_area:
                    logger.warning(f"Skipping small geometry (area={geom.area:.2f}) at index {idx}")
                    continue
                
                # Apply convex hull if configured
                if self.use_convex_hull:
                    geom = geom.convex_hull
                
                # Apply buffer if configured
                if self.buffer_size > 0:
                    geom = geom.buffer(self.buffer_size)
                
                # Create geometry dictionary
                geom_dict = {
                    'id': f"geom_{idx:04d}",
                    'geometry': geom,
                    'bounds': geom.bounds,
                    'area': geom.area,
                    'attributes': row.drop('geometry').to_dict() if 'geometry' in row.index else {}
                }
                
                # Add PNU if available
                if 'PNU' in row.index:
                    geom_dict['pnu'] = str(row['PNU'])
                elif 'pnu' in row.index:
                    geom_dict['pnu'] = str(row['pnu'])
                
                geometries.append(geom_dict)
                
            except Exception as e:
                logger.error(f"Error processing geometry at index {idx}: {str(e)}")
        
        return geometries
    
    def _crop_image_by_geometries(self, image_path: Path, geometries: List[Dict]) -> None:
        """Crop image by multiple geometries
        
        Args:
            image_path: Path to image file
            geometries: List of geometry dictionaries
        """
        try:
            with rasterio.open(image_path) as src:
                # Get image CRS
                img_crs = src.crs
                
                for geom_dict in geometries:
                    try:
                        geom = geom_dict['geometry']
                        geom_id = geom_dict['id']
                        
                        # Transform geometry to image CRS if needed
                        # Assuming both are in same CRS (EPSG:5186) from POD1
                        
                        # Check if geometry intersects with image bounds
                        img_bounds = box(*src.bounds)
                        if not geom.intersects(img_bounds):
                            logger.debug(f"Geometry {geom_id} does not intersect with image")
                            continue
                        
                        # Crop image
                        out_image, out_transform = mask(
                            src, 
                            [geom], 
                            crop=True,
                            nodata=src.nodata
                        )
                        
                        # Skip if cropped image is empty
                        if out_image.size == 0 or np.all(out_image == src.nodata):
                            logger.debug(f"Cropped image for {geom_id} is empty")
                            continue
                        
                        # Create output filename
                        output_filename = f"{image_path.stem}_{geom_id}.tif"
                        if 'pnu' in geom_dict:
                            output_filename = f"{image_path.stem}_PNU_{geom_dict['pnu']}.tif"
                        
                        output_path = self.output_dir / "cropped_images" / output_filename
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Update metadata for cropped image
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "compress": "lzw",
                            "tiled": True
                        })
                        
                        # Write cropped image
                        with rasterio.open(output_path, "w", **out_meta) as dst:
                            dst.write(out_image)
                        
                        # Calculate cropped image statistics
                        crop_stats = self._calculate_image_statistics(out_image)
                        
                        # Add to results
                        crop_result = {
                            'id': f"crop_{len(self.results['cropped_images']):04d}",
                            'source_image': str(image_path),
                            'geometry_id': geom_id,
                            'pnu': geom_dict.get('pnu'),
                            'output_path': str(output_path),
                            'bounds': geom.bounds,
                            'area': geom.area,
                            'image_shape': out_image.shape,
                            'statistics': crop_stats,
                            'processed_at': datetime.now().isoformat()
                        }
                        
                        self.results['cropped_images'].append(crop_result)
                        logger.info(f"Created cropped image: {output_path}")
                        
                    except Exception as e:
                        logger.error(f"Error cropping with geometry {geom_id}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
    
    def _calculate_image_statistics(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for cropped image
        
        Args:
            image_array: Numpy array of image data
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        try:
            # Handle multi-band images
            if len(image_array.shape) == 3:
                for band_idx in range(image_array.shape[0]):
                    band_data = image_array[band_idx]
                    # Filter out nodata values (assuming 0 or negative)
                    valid_data = band_data[band_data > 0]
                    
                    if valid_data.size > 0:
                        band_stats = {
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'mean': float(np.mean(valid_data)),
                            'std': float(np.std(valid_data)),
                            'median': float(np.median(valid_data))
                        }
                    else:
                        band_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'median': 0}
                    
                    stats[f'band_{band_idx + 1}'] = band_stats
            else:
                # Single band image
                valid_data = image_array[image_array > 0]
                if valid_data.size > 0:
                    stats = {
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'median': float(np.median(valid_data))
                    }
                else:
                    stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'median': 0}
                    
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            stats = {}
        
        return stats
    
    def _calculate_statistics(self) -> None:
        """Calculate overall processing statistics"""
        try:
            self.results['statistics'] = {
                'total_cropped_images': len(self.results['cropped_images']),
                'total_area_cropped': sum(
                    img['area'] for img in self.results['cropped_images']
                ),
                'unique_pnus': len(set(
                    img['pnu'] for img in self.results['cropped_images'] 
                    if img.get('pnu')
                )),
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info(f"Statistics: {self.results['statistics']}")
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
    
    def _save_results(self) -> None:
        """Save processing results to JSON"""
        try:
            results_path = self.output_dir / "cropping_results.json"
            self.results['metadata']['updated_at'] = datetime.now().isoformat()
            
            FileUtils.save_json(self.results, results_path)
            logger.info(f"Results saved to: {results_path}")
            
            # Also save cropped regions as GeoJSON
            if self.results['cropped_images']:
                self._save_cropped_regions_geojson()
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _save_cropped_regions_geojson(self) -> None:
        """Save cropped regions as GeoJSON for visualization"""
        try:
            features = []
            
            for crop_info in self.results['cropped_images']:
                # Create feature from bounds
                minx, miny, maxx, maxy = crop_info['bounds']
                geom = box(minx, miny, maxx, maxy)
                
                feature = {
                    'type': 'Feature',
                    'geometry': mapping(geom),
                    'properties': {
                        'id': crop_info['id'],
                        'pnu': crop_info.get('pnu'),
                        'area': crop_info['area'],
                        'source_image': Path(crop_info['source_image']).name,
                        'output_path': Path(crop_info['output_path']).name
                    }
                }
                features.append(feature)
            
            geojson = {
                'type': 'FeatureCollection',
                'features': features,
                'crs': {
                    'type': 'name',
                    'properties': {'name': 'EPSG:5186'}
                }
            }
            
            geojson_path = self.output_dir / "cropped_regions.geojson"
            FileUtils.save_json(geojson, geojson_path)
            logger.info(f"Cropped regions saved as GeoJSON: {geojson_path}")
            
        except Exception as e:
            logger.error(f"Error saving GeoJSON: {str(e)}")