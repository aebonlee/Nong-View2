"""
Metadata extraction utilities for POD1
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import rasterio
from rasterio.crs import CRS
import geopandas as gpd
from osgeo import gdal

from ...core.logger import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """Extract metadata from various file types"""
    
    def extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata from raster image
        
        Args:
            image_path: Path to raster image
            
        Returns:
            Dictionary containing image metadata
        """
        metadata = {}
        
        try:
            with rasterio.open(image_path) as src:
                # Basic information
                metadata['width'] = src.width
                metadata['height'] = src.height
                metadata['bands'] = src.count
                metadata['dtype'] = str(src.dtypes[0])
                metadata['driver'] = src.driver
                
                # Coordinate system
                metadata['crs'] = str(src.crs) if src.crs else None
                
                # Bounds
                metadata['bounds'] = {
                    'minx': src.bounds.left,
                    'miny': src.bounds.bottom,
                    'maxx': src.bounds.right,
                    'maxy': src.bounds.top
                }
                
                # Transform
                metadata['transform'] = src.transform.to_gdal()
                
                # Resolution
                metadata['resolution'] = {
                    'x': abs(src.transform[0]),
                    'y': abs(src.transform[4])
                }
                
                # NoData value
                metadata['nodata'] = src.nodata
                
                # File information
                file_path = Path(image_path)
                metadata['file_size'] = file_path.stat().st_size
                metadata['file_name'] = file_path.name
                metadata['file_format'] = file_path.suffix[1:].upper()
                
                # Tags (if available)
                metadata['tags'] = dict(src.tags())
                
                # Color interpretation
                metadata['color_interpretation'] = [
                    src.colorinterp[i].name if src.colorinterp[i] else 'undefined'
                    for i in range(src.count)
                ]
                
                # Statistics (optional)
                try:
                    stats = []
                    for i in range(1, src.count + 1):
                        band_stats = {
                            'band': i,
                            'min': src.statistics(i).min,
                            'max': src.statistics(i).max,
                            'mean': src.statistics(i).mean,
                            'std': src.statistics(i).std
                        }
                        stats.append(band_stats)
                    metadata['statistics'] = stats
                except:
                    metadata['statistics'] = None
                
                # Capture date (from EXIF if available)
                metadata['capture_date'] = self._extract_capture_date(src.tags())
                
        except Exception as e:
            logger.error(f"Error extracting image metadata: {str(e)}")
            metadata['error'] = str(e)
        
        return metadata
    
    def extract_shapefile_metadata(self, shapefile_path: str) -> Dict[str, Any]:
        """Extract metadata from shapefile
        
        Args:
            shapefile_path: Path to shapefile
            
        Returns:
            Dictionary containing shapefile metadata
        """
        metadata = {}
        
        try:
            gdf = gpd.read_file(shapefile_path)
            
            # Basic information
            metadata['feature_count'] = len(gdf)
            metadata['crs'] = str(gdf.crs) if gdf.crs else None
            metadata['geometry_type'] = gdf.geometry.type.value_counts().to_dict()
            
            # Bounds
            total_bounds = gdf.total_bounds
            metadata['bounds'] = {
                'minx': total_bounds[0],
                'miny': total_bounds[1],
                'maxx': total_bounds[2],
                'maxy': total_bounds[3]
            }
            
            # Columns
            metadata['columns'] = list(gdf.columns)
            metadata['column_types'] = {col: str(dtype) for col, dtype in gdf.dtypes.items()}
            
            # File information
            file_path = Path(shapefile_path)
            metadata['file_name'] = file_path.name
            metadata['file_size'] = file_path.stat().st_size
            
            # Geometry validation
            metadata['geometry_validity'] = {
                'valid': gdf.geometry.is_valid.sum(),
                'invalid': (~gdf.geometry.is_valid).sum()
            }
            
            # Area statistics (for polygons)
            if 'Polygon' in metadata['geometry_type']:
                areas = gdf.geometry.area
                metadata['area_statistics'] = {
                    'total': areas.sum(),
                    'mean': areas.mean(),
                    'min': areas.min(),
                    'max': areas.max()
                }
            
        except Exception as e:
            logger.error(f"Error extracting shapefile metadata: {str(e)}")
            metadata['error'] = str(e)
        
        return metadata
    
    def extract_excel_metadata(self, excel_path: str) -> Dict[str, Any]:
        """Extract metadata from Excel file
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            Dictionary containing Excel metadata
        """
        metadata = {}
        
        try:
            import pandas as pd
            
            # Read Excel file
            df = pd.read_excel(excel_path)
            
            # Basic information
            metadata['row_count'] = len(df)
            metadata['column_count'] = len(df.columns)
            metadata['columns'] = list(df.columns)
            metadata['column_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # File information
            file_path = Path(excel_path)
            metadata['file_name'] = file_path.name
            metadata['file_size'] = file_path.stat().st_size
            
            # Missing values
            metadata['missing_values'] = df.isnull().sum().to_dict()
            
            # Memory usage
            metadata['memory_usage'] = df.memory_usage(deep=True).sum()
            
        except Exception as e:
            logger.error(f"Error extracting Excel metadata: {str(e)}")
            metadata['error'] = str(e)
        
        return metadata
    
    def _extract_capture_date(self, tags: Dict) -> Optional[str]:
        """Extract capture date from image tags
        
        Args:
            tags: Dictionary of image tags
            
        Returns:
            Capture date string or None
        """
        # Common date tags
        date_tags = [
            'TIFFTAG_DATETIME',
            'EXIF_DateTimeOriginal',
            'EXIF_DateTimeDigitized',
            'DateTime'
        ]
        
        for tag in date_tags:
            if tag in tags:
                try:
                    # Parse common date formats
                    date_str = tags[tag]
                    # Try different date formats
                    for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            return dt.isoformat()
                        except:
                            continue
                except:
                    continue
        
        return None
    
    def extract_gdal_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using GDAL (for formats not supported by rasterio)
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary containing GDAL metadata
        """
        metadata = {}
        
        try:
            dataset = gdal.Open(file_path)
            if dataset:
                # Basic information
                metadata['width'] = dataset.RasterXSize
                metadata['height'] = dataset.RasterYSize
                metadata['bands'] = dataset.RasterCount
                
                # Geotransform
                geotransform = dataset.GetGeoTransform()
                if geotransform:
                    metadata['geotransform'] = geotransform
                    metadata['pixel_size'] = {
                        'x': abs(geotransform[1]),
                        'y': abs(geotransform[5])
                    }
                
                # Projection
                projection = dataset.GetProjection()
                if projection:
                    metadata['projection'] = projection
                
                # Metadata
                metadata['metadata'] = dataset.GetMetadata()
                
                # Band information
                bands_info = []
                for i in range(1, dataset.RasterCount + 1):
                    band = dataset.GetRasterBand(i)
                    band_info = {
                        'band': i,
                        'dtype': gdal.GetDataTypeName(band.DataType),
                        'nodata': band.GetNoDataValue(),
                        'min': band.GetMinimum(),
                        'max': band.GetMaximum()
                    }
                    
                    # Statistics
                    stats = band.GetStatistics(False, True)
                    if stats:
                        band_info['statistics'] = {
                            'min': stats[0],
                            'max': stats[1],
                            'mean': stats[2],
                            'std': stats[3]
                        }
                    
                    bands_info.append(band_info)
                
                metadata['bands_info'] = bands_info
                
                dataset = None  # Close dataset
                
        except Exception as e:
            logger.error(f"Error extracting GDAL metadata: {str(e)}")
            metadata['error'] = str(e)
        
        return metadata