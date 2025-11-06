"""
File conversion utilities for POD1
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from osgeo import gdal
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from ...core.logger import get_logger

logger = get_logger(__name__)


class FileConverter:
    """Handle file format conversions"""
    
    def __init__(self):
        """Initialize file converter"""
        # Enable GDAL exceptions
        gdal.UseExceptions()
        
        # Set GDAL configuration options
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
        gdal.SetConfigOption('GDAL_CACHEMAX', '512')  # MB
    
    def convert_ecw_to_tif(self, ecw_path: str, tif_path: str, 
                           compression: str = 'LZW') -> bool:
        """Convert ECW file to GeoTIFF
        
        Args:
            ecw_path: Path to ECW file
            tif_path: Path to output TIF file
            compression: Compression method (LZW, DEFLATE, NONE)
            
        Returns:
            True if conversion successful
        """
        try:
            logger.info(f"Converting ECW to TIF: {ecw_path} -> {tif_path}")
            
            # Ensure output directory exists
            Path(tif_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Open ECW file
            src_ds = gdal.Open(ecw_path)
            if src_ds is None:
                logger.error(f"Failed to open ECW file: {ecw_path}")
                return False
            
            # Create output options
            options = [
                f'COMPRESS={compression}',
                'TILED=YES',
                'BLOCKXSIZE=512',
                'BLOCKYSIZE=512',
                'BIGTIFF=IF_SAFER'
            ]
            
            # Create TIF file
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.CreateCopy(tif_path, src_ds, options=options)
            
            # Flush cache and close
            dst_ds.FlushCache()
            dst_ds = None
            src_ds = None
            
            logger.info(f"Successfully converted ECW to TIF: {tif_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting ECW to TIF: {str(e)}")
            return False
    
    def convert_to_cog(self, input_path: str, output_path: str) -> bool:
        """Convert raster to Cloud Optimized GeoTIFF (COG)
        
        Args:
            input_path: Path to input raster
            output_path: Path to output COG
            
        Returns:
            True if conversion successful
        """
        try:
            logger.info(f"Converting to COG: {input_path} -> {output_path}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Open input file
            src_ds = gdal.Open(input_path)
            if src_ds is None:
                logger.error(f"Failed to open input file: {input_path}")
                return False
            
            # COG creation options
            options = [
                'COMPRESS=LZW',
                'TILED=YES',
                'BLOCKXSIZE=512',
                'BLOCKYSIZE=512',
                'COPY_SRC_OVERVIEWS=YES',
                'BIGTIFF=IF_SAFER'
            ]
            
            # Create COG
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.CreateCopy(output_path, src_ds, options=options)
            
            # Build overviews (pyramids)
            overview_levels = [2, 4, 8, 16, 32]
            dst_ds.BuildOverviews('AVERAGE', overview_levels)
            
            # Flush cache and close
            dst_ds.FlushCache()
            dst_ds = None
            src_ds = None
            
            logger.info(f"Successfully created COG: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating COG: {str(e)}")
            return False
    
    def reproject_raster(self, src_path: str, dst_path: str, 
                        dst_crs: str = "EPSG:5186",
                        resampling_method: str = 'bilinear') -> bool:
        """Reproject raster to target CRS
        
        Args:
            src_path: Source raster path
            dst_path: Destination raster path  
            dst_crs: Target CRS
            resampling_method: Resampling method
            
        Returns:
            True if reprojection successful
        """
        try:
            logger.info(f"Reprojecting raster to {dst_crs}: {src_path} -> {dst_path}")
            
            # Ensure output directory exists
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Map resampling methods
            resampling_map = {
                'nearest': Resampling.nearest,
                'bilinear': Resampling.bilinear,
                'cubic': Resampling.cubic,
                'average': Resampling.average,
                'mode': Resampling.mode
            }
            resampling = resampling_map.get(resampling_method, Resampling.bilinear)
            
            with rasterio.open(src_path) as src:
                # Calculate transform for new CRS
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                
                # Update metadata
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'compress': 'lzw',
                    'tiled': True,
                    'blockxsize': 512,
                    'blockysize': 512
                })
                
                # Reproject
                with rasterio.open(dst_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=resampling
                        )
            
            logger.info(f"Successfully reprojected raster: {dst_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error reprojecting raster: {str(e)}")
            return False
    
    def convert_format(self, input_path: str, output_path: str, 
                      output_format: str = 'GTiff') -> bool:
        """Convert raster between different formats
        
        Args:
            input_path: Input raster path
            output_path: Output raster path
            output_format: Output format (GTiff, HFA, JP2, etc.)
            
        Returns:
            True if conversion successful
        """
        try:
            logger.info(f"Converting format to {output_format}: {input_path} -> {output_path}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Open input file
            src_ds = gdal.Open(input_path)
            if src_ds is None:
                logger.error(f"Failed to open input file: {input_path}")
                return False
            
            # Get driver for output format
            driver = gdal.GetDriverByName(output_format)
            if driver is None:
                logger.error(f"Unsupported output format: {output_format}")
                return False
            
            # Create copy with new format
            dst_ds = driver.CreateCopy(output_path, src_ds)
            
            # Flush cache and close
            dst_ds.FlushCache()
            dst_ds = None
            src_ds = None
            
            logger.info(f"Successfully converted format: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting format: {str(e)}")
            return False
    
    def merge_rasters(self, input_paths: list, output_path: str) -> bool:
        """Merge multiple rasters into one
        
        Args:
            input_paths: List of input raster paths
            output_path: Output merged raster path
            
        Returns:
            True if merge successful
        """
        try:
            logger.info(f"Merging {len(input_paths)} rasters to: {output_path}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Build VRT (Virtual Raster)
            vrt_path = output_path.replace('.tif', '.vrt')
            vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
            vrt_ds = gdal.BuildVRT(vrt_path, input_paths, options=vrt_options)
            
            if vrt_ds is None:
                logger.error("Failed to build VRT")
                return False
            
            # Convert VRT to TIF
            driver = gdal.GetDriverByName('GTiff')
            options = [
                'COMPRESS=LZW',
                'TILED=YES',
                'BIGTIFF=IF_SAFER'
            ]
            
            dst_ds = driver.CreateCopy(output_path, vrt_ds, options=options)
            
            # Cleanup
            dst_ds = None
            vrt_ds = None
            
            # Remove temporary VRT file
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            
            logger.info(f"Successfully merged rasters: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging rasters: {str(e)}")
            return False
    
    def extract_band(self, input_path: str, output_path: str, 
                    band_number: int = 1) -> bool:
        """Extract specific band from multi-band raster
        
        Args:
            input_path: Input raster path
            output_path: Output raster path
            band_number: Band number to extract (1-based)
            
        Returns:
            True if extraction successful
        """
        try:
            logger.info(f"Extracting band {band_number}: {input_path} -> {output_path}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Open input file
            src_ds = gdal.Open(input_path)
            if src_ds is None:
                logger.error(f"Failed to open input file: {input_path}")
                return False
            
            # Check band number
            if band_number < 1 or band_number > src_ds.RasterCount:
                logger.error(f"Invalid band number: {band_number}")
                return False
            
            # Get band
            band = src_ds.GetRasterBand(band_number)
            
            # Create output file
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                output_path,
                src_ds.RasterXSize,
                src_ds.RasterYSize,
                1,  # Single band
                band.DataType,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            
            # Copy georeferencing
            dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
            dst_ds.SetProjection(src_ds.GetProjection())
            
            # Copy band data
            dst_band = dst_ds.GetRasterBand(1)
            dst_band.WriteArray(band.ReadAsArray())
            
            # Copy nodata value if exists
            nodata = band.GetNoDataValue()
            if nodata is not None:
                dst_band.SetNoDataValue(nodata)
            
            # Flush and close
            dst_band.FlushCache()
            dst_ds = None
            src_ds = None
            
            logger.info(f"Successfully extracted band: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting band: {str(e)}")
            return False