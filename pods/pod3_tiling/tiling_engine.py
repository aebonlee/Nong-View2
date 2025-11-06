"""
POD3: Tiling engine for splitting images into tiles
"""
import os
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box, mapping

from ...core.config import get_config
from ...core.logger import get_logger
from ...core.utils import FileUtils

logger = get_logger(__name__)


class TilingEngine:
    """Main engine for POD3 tiling operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize tiling engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_config().get('tiling')
        
        # Setup directories
        self.input_dir = get_config().output_dir / "pod2_output"
        self.output_dir = get_config().output_dir / "pod3_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tiling parameters
        self.tile_size = self.config.get('tile_size', 1024)
        self.overlap = self.config.get('overlap', 0.2)  # 20% overlap
        self.output_format = self.config.get('output_format', 'tif')
        self.adaptive_tiling = self.config.get('adaptive_tiling', True)
        self.min_tile_size = self.config.get('min_tile_size', 512)
        self.max_tile_size = self.config.get('max_tile_size', 2048)
        
        # Initialize results registry
        self.results = {
            'tiles': [],
            'tile_index': {},
            'statistics': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'tile_size': self.tile_size,
                'overlap': self.overlap
            }
        }
        
        logger.info(f"POD3 Tiling Engine initialized (tile_size={self.tile_size}, overlap={self.overlap})")
    
    def process(self, image_source: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Main processing method for tiling
        
        Args:
            image_source: Path to image or directory (uses POD2 output if None)
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting POD3 tiling process")
        
        try:
            # Get images to process
            if image_source is None:
                # Use POD2 output
                image_source = self.input_dir / "cropped_images"
                if not image_source.exists():
                    # Fall back to POD2 results file
                    results_file = self.input_dir / "cropping_results.json"
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            pod2_results = json.load(f)
                            images = [
                                Path(img['output_path']) 
                                for img in pod2_results.get('cropped_images', [])
                                if Path(img['output_path']).exists()
                            ]
                    else:
                        images = []
                else:
                    images = self._get_image_list(image_source)
            else:
                images = self._get_image_list(image_source)
            
            logger.info(f"Found {len(images)} images to tile")
            
            # Process each image
            for img_path in images:
                logger.info(f"Tiling image: {img_path}")
                self._tile_image(img_path)
            
            # Create spatial index
            self._create_spatial_index()
            
            # Calculate statistics
            self._calculate_statistics()
            
            # Save results
            self._save_results()
            
            self.results['status'] = 'completed'
            self.results['output_path'] = str(self.output_dir)
            
            logger.info(f"POD3 processing completed. Output: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error in POD3 processing: {str(e)}")
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
        source = Path(source)
        
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
    
    def _tile_image(self, image_path: Path) -> None:
        """Tile a single image
        
        Args:
            image_path: Path to image file
        """
        try:
            with rasterio.open(image_path) as src:
                # Get image dimensions
                img_width = src.width
                img_height = src.height
                
                # Determine tile size (adaptive if configured)
                tile_width, tile_height = self._determine_tile_size(img_width, img_height)
                
                # Calculate overlap in pixels
                overlap_x = int(tile_width * self.overlap)
                overlap_y = int(tile_height * self.overlap)
                
                # Calculate stride (step size)
                stride_x = tile_width - overlap_x
                stride_y = tile_height - overlap_y
                
                # Calculate number of tiles
                n_tiles_x = math.ceil((img_width - overlap_x) / stride_x)
                n_tiles_y = math.ceil((img_height - overlap_y) / stride_y)
                
                logger.info(f"Creating {n_tiles_x}x{n_tiles_y} tiles from {image_path.name}")
                
                # Create output directory for this image's tiles
                image_stem = image_path.stem
                tiles_dir = self.output_dir / "tiles" / image_stem
                tiles_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate tiles
                tile_count = 0
                for row in range(n_tiles_y):
                    for col in range(n_tiles_x):
                        # Calculate window position
                        x_off = col * stride_x
                        y_off = row * stride_y
                        
                        # Adjust window size for edge tiles
                        win_width = min(tile_width, img_width - x_off)
                        win_height = min(tile_height, img_height - y_off)
                        
                        # Skip if tile is too small
                        if win_width < self.min_tile_size or win_height < self.min_tile_size:
                            continue
                        
                        # Create window
                        window = Window(x_off, y_off, win_width, win_height)
                        
                        # Read tile data
                        tile_data = src.read(window=window)
                        
                        # Skip empty tiles (all zeros or nodata)
                        if np.all(tile_data == src.nodata) or np.all(tile_data == 0):
                            continue
                        
                        # Calculate transform for this tile
                        tile_transform = rasterio.windows.transform(window, src.transform)
                        
                        # Create output filename
                        tile_filename = f"{image_stem}_tile_{row:03d}_{col:03d}.{self.output_format}"
                        tile_path = tiles_dir / tile_filename
                        
                        # Create tile metadata
                        tile_meta = src.meta.copy()
                        tile_meta.update({
                            'driver': 'GTiff' if self.output_format == 'tif' else self.output_format.upper(),
                            'height': win_height,
                            'width': win_width,
                            'transform': tile_transform,
                            'compress': 'lzw',
                            'tiled': True,
                            'blockxsize': 256,
                            'blockysize': 256
                        })
                        
                        # Write tile
                        with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                            dst.write(tile_data)
                        
                        # Calculate tile bounds
                        tile_bounds = rasterio.windows.bounds(window, src.transform)
                        
                        # Add tile info to results
                        tile_info = {
                            'id': f"tile_{len(self.results['tiles']):05d}",
                            'source_image': str(image_path),
                            'tile_path': str(tile_path),
                            'row': row,
                            'col': col,
                            'x_offset': x_off,
                            'y_offset': y_off,
                            'width': win_width,
                            'height': win_height,
                            'bounds': tile_bounds,
                            'transform': tile_transform.to_gdal(),
                            'processed_at': datetime.now().isoformat()
                        }
                        
                        self.results['tiles'].append(tile_info)
                        
                        # Add to tile index
                        if image_stem not in self.results['tile_index']:
                            self.results['tile_index'][image_stem] = []
                        self.results['tile_index'][image_stem].append(tile_info['id'])
                        
                        tile_count += 1
                
                logger.info(f"Created {tile_count} tiles from {image_path.name}")
                
        except Exception as e:
            logger.error(f"Error tiling image {image_path}: {str(e)}")
    
    def _determine_tile_size(self, img_width: int, img_height: int) -> Tuple[int, int]:
        """Determine optimal tile size based on image dimensions
        
        Args:
            img_width: Image width
            img_height: Image height
            
        Returns:
            Tuple of (tile_width, tile_height)
        """
        if not self.adaptive_tiling:
            return self.tile_size, self.tile_size
        
        # Start with configured tile size
        tile_width = self.tile_size
        tile_height = self.tile_size
        
        # Adjust if image is smaller than tile size
        if img_width < tile_width:
            tile_width = max(img_width, self.min_tile_size)
        if img_height < tile_height:
            tile_height = max(img_height, self.min_tile_size)
        
        # For very large images, consider larger tiles
        if img_width > 10000 or img_height > 10000:
            # Use larger tiles for efficiency
            tile_width = min(self.max_tile_size, tile_width * 2)
            tile_height = min(self.max_tile_size, tile_height * 2)
        
        # Try to minimize the number of small edge tiles
        # by slightly adjusting tile size
        if self.adaptive_tiling:
            # Calculate how many tiles we'd get
            overlap_x = int(tile_width * self.overlap)
            overlap_y = int(tile_height * self.overlap)
            stride_x = tile_width - overlap_x
            stride_y = tile_height - overlap_y
            
            n_tiles_x = math.ceil((img_width - overlap_x) / stride_x)
            n_tiles_y = math.ceil((img_height - overlap_y) / stride_y)
            
            # Check if last tiles would be very small
            last_tile_width = img_width - (n_tiles_x - 1) * stride_x
            last_tile_height = img_height - (n_tiles_y - 1) * stride_y
            
            # If last tiles are too small, adjust tile size
            if last_tile_width < self.min_tile_size and n_tiles_x > 1:
                # Distribute the width more evenly
                tile_width = int(img_width / (n_tiles_x - 0.5))
                tile_width = min(max(tile_width, self.min_tile_size), self.max_tile_size)
            
            if last_tile_height < self.min_tile_size and n_tiles_y > 1:
                # Distribute the height more evenly
                tile_height = int(img_height / (n_tiles_y - 0.5))
                tile_height = min(max(tile_height, self.min_tile_size), self.max_tile_size)
        
        return tile_width, tile_height
    
    def _create_spatial_index(self) -> None:
        """Create spatial index for tiles"""
        try:
            if not self.results['tiles']:
                return
            
            # Create GeoDataFrame from tiles
            features = []
            for tile in self.results['tiles']:
                # Create polygon from bounds
                minx, miny, maxx, maxy = tile['bounds']
                geom = box(minx, miny, maxx, maxy)
                
                features.append({
                    'geometry': geom,
                    'id': tile['id'],
                    'source': Path(tile['source_image']).name,
                    'tile_path': Path(tile['tile_path']).name,
                    'row': tile['row'],
                    'col': tile['col']
                })
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(features, crs='EPSG:5186')
            
            # Save as GeoPackage for spatial indexing
            index_path = self.output_dir / "tile_index.gpkg"
            gdf.to_file(index_path, driver='GPKG', layer='tiles')
            
            logger.info(f"Spatial index created: {index_path}")
            
            # Also save as GeoJSON for visualization
            geojson_path = self.output_dir / "tile_index.geojson"
            gdf.to_file(geojson_path, driver='GeoJSON')
            
        except Exception as e:
            logger.error(f"Error creating spatial index: {str(e)}")
    
    def _calculate_statistics(self) -> None:
        """Calculate tiling statistics"""
        try:
            total_tiles = len(self.results['tiles'])
            unique_sources = len(set(tile['source_image'] for tile in self.results['tiles']))
            
            # Calculate tile size distribution
            tile_sizes = [(tile['width'], tile['height']) for tile in self.results['tiles']]
            unique_sizes = set(tile_sizes)
            
            # Calculate coverage area
            total_area = 0
            for tile in self.results['tiles']:
                minx, miny, maxx, maxy = tile['bounds']
                area = (maxx - minx) * (maxy - miny)
                total_area += area
            
            self.results['statistics'] = {
                'total_tiles': total_tiles,
                'unique_sources': unique_sources,
                'unique_tile_sizes': len(unique_sizes),
                'tile_sizes': list(unique_sizes),
                'total_coverage_area': total_area,
                'average_tiles_per_image': total_tiles / unique_sources if unique_sources > 0 else 0,
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info(f"Statistics: {self.results['statistics']}")
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
    
    def _save_results(self) -> None:
        """Save tiling results to JSON"""
        try:
            results_path = self.output_dir / "tiling_results.json"
            self.results['metadata']['updated_at'] = datetime.now().isoformat()
            
            # Save full results
            FileUtils.save_json(self.results, results_path)
            logger.info(f"Results saved to: {results_path}")
            
            # Save lightweight tile manifest
            manifest = {
                'metadata': self.results['metadata'],
                'statistics': self.results['statistics'],
                'tile_index': self.results['tile_index'],
                'tile_count': len(self.results['tiles'])
            }
            
            manifest_path = self.output_dir / "tile_manifest.json"
            FileUtils.save_json(manifest, manifest_path)
            logger.info(f"Tile manifest saved to: {manifest_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")