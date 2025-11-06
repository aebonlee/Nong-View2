"""
Common utilities for Nong-View2
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from osgeo import gdal, osr
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box, Polygon, MultiPolygon
import geopandas as gpd


class GeoUtils:
    """Geospatial utilities"""
    
    @staticmethod
    def get_raster_info(raster_path: str) -> Dict[str, Any]:
        """Get raster metadata information
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            Dictionary containing raster metadata
        """
        with rasterio.open(raster_path) as src:
            info = {
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': str(src.dtypes[0]),
                'crs': str(src.crs),
                'transform': src.transform.to_gdal(),
                'bounds': src.bounds,
                'nodata': src.nodata
            }
        return info
    
    @staticmethod
    def convert_ecw_to_tif(ecw_path: str, tif_path: str) -> bool:
        """Convert ECW file to TIF format
        
        Args:
            ecw_path: Path to ECW file
            tif_path: Path to output TIF file
            
        Returns:
            True if conversion successful
        """
        try:
            # Open ECW file
            src_ds = gdal.Open(ecw_path)
            if src_ds is None:
                return False
            
            # Create TIF file with compression
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.CreateCopy(
                tif_path, 
                src_ds,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            )
            
            # Close datasets
            dst_ds = None
            src_ds = None
            
            return True
        except Exception as e:
            print(f"Error converting ECW to TIF: {e}")
            return False
    
    @staticmethod
    def validate_crs(file_path: str, target_crs: str = "EPSG:5186") -> bool:
        """Validate if file has correct CRS
        
        Args:
            file_path: Path to geospatial file
            target_crs: Target CRS to validate against
            
        Returns:
            True if CRS matches target
        """
        try:
            with rasterio.open(file_path) as src:
                return src.crs == CRS.from_string(target_crs)
        except:
            return False
    
    @staticmethod
    def reproject_raster(src_path: str, dst_path: str, dst_crs: str = "EPSG:5186") -> bool:
        """Reproject raster to target CRS
        
        Args:
            src_path: Source raster path
            dst_path: Destination raster path
            dst_crs: Target CRS
            
        Returns:
            True if reprojection successful
        """
        try:
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            
            with rasterio.open(src_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                with rasterio.open(dst_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear
                        )
            return True
        except Exception as e:
            print(f"Error reprojecting raster: {e}")
            return False
    
    @staticmethod
    def create_bounds_polygon(bounds: tuple) -> Polygon:
        """Create polygon from bounds
        
        Args:
            bounds: (minx, miny, maxx, maxy)
            
        Returns:
            Shapely Polygon
        """
        minx, miny, maxx, maxy = bounds
        return box(minx, miny, maxx, maxy)


class FileUtils:
    """File handling utilities"""
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Calculate file hash (MD5)
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """Ensure directory exists
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def save_json(data: Any, file_path: Union[str, Path]) -> None:
        """Save data to JSON file
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Any:
        """Load data from JSON file
        
        Args:
            file_path: JSON file path
            
        Returns:
            Loaded data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def list_files(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
        """List files in directory with optional extension filter
        
        Args:
            directory: Directory path
            extensions: List of file extensions to filter
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        if not directory.exists():
            return []
        
        files = []
        for file in directory.iterdir():
            if file.is_file():
                if extensions is None or file.suffix.lower()[1:] in extensions:
                    files.append(file)
        
        return sorted(files)


class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_image(image_path: str) -> Tuple[bool, str]:
        """Validate image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return False, "File does not exist"
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                return False, "File is empty"
            
            # Try to open with rasterio
            with rasterio.open(image_path) as src:
                if src.width == 0 or src.height == 0:
                    return False, "Image has zero dimensions"
                
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    @staticmethod
    def validate_shapefile(shp_path: str) -> Tuple[bool, str]:
        """Validate shapefile
        
        Args:
            shp_path: Path to shapefile
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check if file exists
            if not os.path.exists(shp_path):
                return False, "File does not exist"
            
            # Try to read with geopandas
            gdf = gpd.read_file(shp_path)
            
            if len(gdf) == 0:
                return False, "Shapefile is empty"
            
            if gdf.crs is None:
                return False, "Shapefile has no CRS"
            
            # Check geometry validity
            invalid_geoms = ~gdf.geometry.is_valid
            if invalid_geoms.any():
                return False, f"Shapefile contains {invalid_geoms.sum()} invalid geometries"
            
            return True, "Valid shapefile"
            
        except Exception as e:
            return False, f"Error validating shapefile: {str(e)}"
    
    @staticmethod
    def validate_pnu(pnu: str) -> bool:
        """Validate PNU (Parcel Number) format
        
        Args:
            pnu: PNU string (19 digits)
            
        Returns:
            True if valid PNU format
        """
        if not isinstance(pnu, str):
            pnu = str(pnu)
        
        # Remove any non-digit characters
        pnu = ''.join(filter(str.isdigit, pnu))
        
        # PNU should be 19 digits
        return len(pnu) == 19


class ProcessingUtils:
    """Data processing utilities"""
    
    @staticmethod
    def calculate_statistics(array: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for numpy array
        
        Args:
            array: Numpy array
            
        Returns:
            Dictionary of statistics
        """
        return {
            'min': float(np.min(array)),
            'max': float(np.max(array)),
            'mean': float(np.mean(array)),
            'std': float(np.std(array)),
            'median': float(np.median(array))
        }
    
    @staticmethod
    def clip_raster_by_polygon(raster_path: str, polygon: Polygon, output_path: str) -> bool:
        """Clip raster by polygon
        
        Args:
            raster_path: Input raster path
            polygon: Clipping polygon
            output_path: Output raster path
            
        Returns:
            True if successful
        """
        try:
            from rasterio.mask import mask
            
            with rasterio.open(raster_path) as src:
                out_image, out_transform = mask(src, [polygon], crop=True)
                out_meta = src.meta.copy()
                
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)
            
            return True
            
        except Exception as e:
            print(f"Error clipping raster: {e}")
            return False