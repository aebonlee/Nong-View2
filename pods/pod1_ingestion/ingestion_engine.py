"""
POD1: Main ingestion engine for data import and management
"""
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, shape

from ...core.config import get_config
from ...core.logger import get_logger
from ...core.utils import FileUtils, GeoUtils, ValidationUtils
from .metadata_extractor import MetadataExtractor
from .file_converter import FileConverter

logger = get_logger(__name__)


class IngestionEngine:
    """Main engine for POD1 data ingestion"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize ingestion engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_config().get('ingestion')
        self.metadata_extractor = MetadataExtractor()
        self.file_converter = FileConverter()
        
        # Setup directories
        self.input_dir = get_config().input_dir
        self.output_dir = get_config().output_dir / "pod1_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry
        self.registry = {
            'images': {},
            'shapefiles': {},
            'parcels': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        logger.info("POD1 Ingestion Engine initialized")
    
    def process(self, 
                image_path: Optional[Union[str, Path]] = None,
                shapefile_path: Optional[Union[str, Path]] = None,
                excel_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Main processing method for data ingestion
        
        Args:
            image_path: Path to orthophoto image (TIF/ECW)
            shapefile_path: Path to analysis area shapefile
            excel_path: Path to Excel file with PNU data
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting POD1 data ingestion process")
        results = {
            'status': 'processing',
            'images': [],
            'shapefiles': [],
            'parcels': [],
            'errors': []
        }
        
        try:
            # Process orthophoto images
            if image_path:
                image_results = self._process_images(image_path)
                results['images'] = image_results
            
            # Process shapefiles
            if shapefile_path:
                shapefile_results = self._process_shapefile(shapefile_path)
                results['shapefiles'] = shapefile_results
            
            # Process Excel with PNU data
            if excel_path:
                parcel_results = self._process_excel(excel_path)
                results['parcels'] = parcel_results
            
            # Validate and cross-reference data
            validation_results = self._validate_data()
            results['validation'] = validation_results
            
            # Save registry
            self._save_registry()
            
            results['status'] = 'completed'
            results['output_path'] = str(self.output_dir)
            
            logger.info(f"POD1 processing completed. Output: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error in POD1 processing: {str(e)}")
            results['status'] = 'error'
            results['errors'].append(str(e))
        
        return results
    
    def _process_images(self, image_path: Union[str, Path]) -> List[Dict]:
        """Process orthophoto images
        
        Args:
            image_path: Path to image or directory
            
        Returns:
            List of processed image information
        """
        image_path = Path(image_path)
        processed_images = []
        
        if image_path.is_file():
            image_files = [image_path]
        else:
            # List all supported image files in directory
            supported_formats = self.config.get('supported_formats', ['tif', 'tiff', 'ecw'])
            image_files = FileUtils.list_files(image_path, supported_formats)
        
        for img_file in image_files:
            logger.info(f"Processing image: {img_file}")
            
            try:
                # Convert ECW to TIF if needed
                if img_file.suffix.lower() == '.ecw' and self.config.get('ecw_to_tif', True):
                    tif_path = self.output_dir / "converted" / f"{img_file.stem}.tif"
                    tif_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if self.file_converter.convert_ecw_to_tif(str(img_file), str(tif_path)):
                        logger.info(f"Converted ECW to TIF: {tif_path}")
                        img_file = tif_path
                    else:
                        logger.error(f"Failed to convert ECW: {img_file}")
                        continue
                
                # Validate image
                is_valid, message = ValidationUtils.validate_image(str(img_file))
                if not is_valid:
                    logger.warning(f"Invalid image {img_file}: {message}")
                    continue
                
                # Extract metadata
                metadata = self.metadata_extractor.extract_image_metadata(str(img_file))
                
                # Validate CRS
                target_crs = self.config.get('target_crs', 'EPSG:5186')
                if self.config.get('validate_crs', True):
                    if metadata['crs'] != target_crs:
                        logger.info(f"Reprojecting image to {target_crs}")
                        reprojected_path = self.output_dir / "reprojected" / img_file.name
                        reprojected_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if GeoUtils.reproject_raster(str(img_file), str(reprojected_path), target_crs):
                            img_file = reprojected_path
                            metadata = self.metadata_extractor.extract_image_metadata(str(img_file))
                
                # Copy to output directory
                output_path = self.output_dir / "images" / img_file.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, output_path)
                
                # Add to registry
                image_info = {
                    'id': f"img_{len(processed_images):04d}",
                    'original_path': str(img_file),
                    'output_path': str(output_path),
                    'metadata': metadata,
                    'processed_at': datetime.now().isoformat()
                }
                
                self.registry['images'][image_info['id']] = image_info
                processed_images.append(image_info)
                
            except Exception as e:
                logger.error(f"Error processing image {img_file}: {str(e)}")
        
        return processed_images
    
    def _process_shapefile(self, shapefile_path: Union[str, Path]) -> List[Dict]:
        """Process shapefile with analysis areas
        
        Args:
            shapefile_path: Path to shapefile
            
        Returns:
            List of processed shapefile information
        """
        shapefile_path = Path(shapefile_path)
        processed_shapes = []
        
        try:
            # Validate shapefile
            is_valid, message = ValidationUtils.validate_shapefile(str(shapefile_path))
            if not is_valid:
                logger.error(f"Invalid shapefile: {message}")
                return processed_shapes
            
            # Read shapefile
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Loaded shapefile with {len(gdf)} features")
            
            # Reproject if needed
            target_crs = self.config.get('target_crs', 'EPSG:5186')
            if str(gdf.crs) != target_crs:
                logger.info(f"Reprojecting shapefile to {target_crs}")
                gdf = gdf.to_crs(target_crs)
            
            # Process each feature
            for idx, row in gdf.iterrows():
                try:
                    # Extract PNU if available
                    pnu = None
                    pnu_column = self.config.get('excel_pnu_column', 'PNU')
                    if pnu_column in row:
                        pnu = str(row[pnu_column])
                        if not ValidationUtils.validate_pnu(pnu):
                            logger.warning(f"Invalid PNU format: {pnu}")
                    
                    # Extract geometry
                    geom = row.geometry
                    
                    # Apply convex hull if configured
                    if self.config.get('use_convex_hull', True) and geom:
                        geom = geom.convex_hull
                    
                    # Create feature info
                    feature_info = {
                        'id': f"shp_{idx:04d}",
                        'pnu': pnu,
                        'geometry': mapping(geom),
                        'bounds': geom.bounds if geom else None,
                        'area': geom.area if geom else 0,
                        'attributes': row.drop('geometry').to_dict()
                    }
                    
                    processed_shapes.append(feature_info)
                    
                except Exception as e:
                    logger.error(f"Error processing shapefile feature {idx}: {str(e)}")
            
            # Save processed shapefile
            output_path = self.output_dir / "shapefiles" / f"processed_{shapefile_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_path)
            
            # Add to registry
            shapefile_info = {
                'id': f"shp_collection_{len(self.registry['shapefiles']):04d}",
                'original_path': str(shapefile_path),
                'output_path': str(output_path),
                'features': processed_shapes,
                'crs': str(gdf.crs),
                'total_features': len(processed_shapes),
                'processed_at': datetime.now().isoformat()
            }
            
            self.registry['shapefiles'][shapefile_info['id']] = shapefile_info
            
        except Exception as e:
            logger.error(f"Error processing shapefile: {str(e)}")
        
        return processed_shapes
    
    def _process_excel(self, excel_path: Union[str, Path]) -> List[Dict]:
        """Process Excel file with PNU and address data
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            List of processed parcel information
        """
        excel_path = Path(excel_path)
        processed_parcels = []
        
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            logger.info(f"Loaded Excel with {len(df)} rows")
            
            pnu_column = self.config.get('excel_pnu_column', 'PNU')
            address_column = self.config.get('excel_address_column', '지번')
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    pnu = str(row.get(pnu_column, ''))
                    address = str(row.get(address_column, ''))
                    
                    # Validate PNU
                    if ValidationUtils.validate_pnu(pnu):
                        parcel_info = {
                            'id': f"parcel_{idx:04d}",
                            'pnu': pnu,
                            'address': address,
                            'attributes': row.to_dict()
                        }
                        processed_parcels.append(parcel_info)
                    else:
                        logger.warning(f"Invalid PNU at row {idx}: {pnu}")
                        
                except Exception as e:
                    logger.error(f"Error processing Excel row {idx}: {str(e)}")
            
            # Save processed data
            output_path = self.output_dir / "parcels" / "parcels.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            FileUtils.save_json(processed_parcels, output_path)
            
            # Add to registry
            self.registry['parcels'] = {
                'source': str(excel_path),
                'output_path': str(output_path),
                'total_parcels': len(processed_parcels),
                'parcels': processed_parcels,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
        
        return processed_parcels
    
    def _validate_data(self) -> Dict[str, Any]:
        """Validate and cross-reference all ingested data
        
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Check if we have required data
            if not self.registry['images']:
                validation_results['errors'].append("No images found")
                validation_results['is_valid'] = False
            
            if not self.registry['shapefiles'] and not self.registry['parcels']:
                validation_results['warnings'].append("No spatial boundaries found")
            
            # Calculate statistics
            validation_results['statistics'] = {
                'total_images': len(self.registry['images']),
                'total_shapefiles': len(self.registry['shapefiles']),
                'total_parcels': len(self.registry['parcels'].get('parcels', [])),
                'total_features': sum(
                    len(shp['features']) 
                    for shp in self.registry['shapefiles'].values()
                )
            }
            
            logger.info(f"Validation completed: {validation_results['statistics']}")
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            validation_results['errors'].append(str(e))
            validation_results['is_valid'] = False
        
        return validation_results
    
    def _save_registry(self) -> None:
        """Save registry to JSON file"""
        try:
            registry_path = self.output_dir / "registry.json"
            self.registry['metadata']['updated_at'] = datetime.now().isoformat()
            FileUtils.save_json(self.registry, registry_path)
            logger.info(f"Registry saved to: {registry_path}")
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")