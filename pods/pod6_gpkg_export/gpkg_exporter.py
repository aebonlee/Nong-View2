"""
POD6: GPKG Exporter for final output generation
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

from ...core.config import get_config
from ...core.logger import get_logger
from ...core.utils import FileUtils

logger = get_logger(__name__)


class GPKGExporter:
    """Main engine for POD6 GPKG export operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize GPKG exporter
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_config().get('gpkg_export')
        
        # Setup directories
        self.input_dir = get_config().output_dir / "pod5_output"
        self.output_dir = get_config().output_dir / "pod6_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export parameters
        self.output_format = self.config.get('output_format', 'gpkg')
        self.coordinate_system = self.config.get('coordinate_system', 'EPSG:5186')
        self.calculate_area = self.config.get('calculate_area', True)
        self.generate_report = self.config.get('generate_report', True)
        self.capture_screenshots = self.config.get('capture_screenshots', True)
        self.include_metadata = self.config.get('include_metadata', True)
        
        # Layer configurations
        self.layers = self.config.get('layers', [
            {'name': 'parcels', 'geometry': 'polygon'},
            {'name': 'detections', 'geometry': 'polygon'},
            {'name': 'statistics', 'geometry': 'none'}
        ])
        
        # Initialize results
        self.results = {
            'exported_files': [],
            'layers': {},
            'statistics': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'crs': self.coordinate_system
            }
        }
        
        logger.info("POD6 GPKG Exporter initialized")
    
    def process(self, detection_source: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Main processing method for GPKG export
        
        Args:
            detection_source: Path to merged detections (uses POD5 output if None)
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting POD6 GPKG export process")
        
        try:
            # Load merged detections
            detections = self._load_merged_detections(detection_source)
            logger.info(f"Loaded {len(detections)} merged detections")
            
            # Load parcel data if available
            parcels = self._load_parcel_data()
            
            # Create GeoDataFrames
            detection_gdf = self._create_detection_gdf(detections)
            parcel_gdf = self._create_parcel_gdf(parcels) if parcels else None
            
            # Calculate areas if configured
            if self.calculate_area:
                detection_gdf = self._calculate_areas(detection_gdf)
                if parcel_gdf is not None:
                    parcel_gdf = self._calculate_areas(parcel_gdf)
            
            # Clip detections to parcels if both available
            if parcel_gdf is not None and not detection_gdf.empty:
                clipped_gdf = self._clip_to_parcels(detection_gdf, parcel_gdf)
            else:
                clipped_gdf = detection_gdf
            
            # Create statistics
            statistics_df = self._create_statistics(clipped_gdf, parcel_gdf)
            
            # Export to GPKG
            gpkg_path = self._export_gpkg(detection_gdf, parcel_gdf, clipped_gdf, statistics_df)
            
            # Generate visualizations if configured
            if self.capture_screenshots:
                self._generate_visualizations(detection_gdf, parcel_gdf)
            
            # Generate report if configured
            if self.generate_report:
                report_path = self._generate_report(statistics_df)
                self.results['report_path'] = str(report_path)
            
            # Save results
            self._save_results()
            
            self.results['status'] = 'completed'
            self.results['output_path'] = str(self.output_dir)
            self.results['gpkg_path'] = str(gpkg_path)
            
            logger.info(f"POD6 processing completed. Output: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error in POD6 processing: {str(e)}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
        
        return self.results
    
    def _load_merged_detections(self, source: Optional[Union[str, Path]]) -> List[Dict]:
        """Load merged detections from POD5
        
        Args:
            source: Detection source path
            
        Returns:
            List of merged detections
        """
        detections = []
        
        if source is None:
            # Use POD5 output
            results_file = self.input_dir / "merging_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    pod5_results = json.load(f)
                    detections = pod5_results.get('merged_detections', [])
        else:
            # Load from provided source
            source = Path(source)
            if source.is_file():
                with open(source, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'merged_detections' in data:
                        detections = data['merged_detections']
                    elif isinstance(data, list):
                        detections = data
        
        return detections
    
    def _load_parcel_data(self) -> Optional[List[Dict]]:
        """Load parcel data from POD1
        
        Returns:
            List of parcels or None
        """
        try:
            # Try to load from POD1 output
            pod1_dir = get_config().output_dir / "pod1_output"
            
            # Check for parcels.json
            parcels_file = pod1_dir / "parcels" / "parcels.json"
            if parcels_file.exists():
                with open(parcels_file, 'r') as f:
                    return json.load(f)
            
            # Check for processed shapefiles
            shp_dir = pod1_dir / "shapefiles"
            if shp_dir.exists():
                for shp_file in shp_dir.glob("*.shp"):
                    gdf = gpd.read_file(shp_file)
                    return gdf.to_dict('records')
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not load parcel data: {str(e)}")
            return None
    
    def _create_detection_gdf(self, detections: List[Dict]) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            GeoDataFrame of detections
        """
        if not detections:
            return gpd.GeoDataFrame()
        
        features = []
        for detection in detections:
            try:
                # Get geometry
                if 'geometry' in detection:
                    if isinstance(detection['geometry'], dict):
                        geom = shape(detection['geometry'])
                    else:
                        geom = detection['geometry']
                elif 'bbox' in detection:
                    bbox = detection['bbox']
                    geom = Polygon([
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]]
                    ])
                else:
                    continue
                
                # Create feature
                feature = {
                    'geometry': geom,
                    'id': detection.get('id'),
                    'class_name': detection.get('class_name'),
                    'class_id': detection.get('class_id'),
                    'confidence': detection.get('confidence'),
                    'merged_count': detection.get('merged_count', 1)
                }
                
                features.append(feature)
                
            except Exception as e:
                logger.error(f"Error creating feature from detection: {str(e)}")
        
        if features:
            gdf = gpd.GeoDataFrame(features, crs=self.coordinate_system)
            return gdf
        else:
            return gpd.GeoDataFrame()
    
    def _create_parcel_gdf(self, parcels: List[Dict]) -> Optional[gpd.GeoDataFrame]:
        """Create GeoDataFrame from parcels
        
        Args:
            parcels: List of parcel dictionaries
            
        Returns:
            GeoDataFrame of parcels or None
        """
        if not parcels:
            return None
        
        features = []
        for parcel in parcels:
            try:
                feature = {
                    'id': parcel.get('id'),
                    'pnu': parcel.get('pnu'),
                    'address': parcel.get('address')
                }
                
                # Add geometry if available
                if 'geometry' in parcel:
                    if isinstance(parcel['geometry'], dict):
                        feature['geometry'] = shape(parcel['geometry'])
                    else:
                        feature['geometry'] = parcel['geometry']
                
                # Add other attributes
                for key, value in parcel.get('attributes', {}).items():
                    if key not in feature:
                        feature[key] = value
                
                features.append(feature)
                
            except Exception as e:
                logger.error(f"Error creating feature from parcel: {str(e)}")
        
        if features:
            # Check if geometries exist
            if any('geometry' in f for f in features):
                gdf = gpd.GeoDataFrame(features, crs=self.coordinate_system)
            else:
                # Create DataFrame without geometry
                gdf = gpd.GeoDataFrame(features)
            return gdf
        else:
            return None
    
    def _calculate_areas(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate areas for geometries
        
        Args:
            gdf: GeoDataFrame
            
        Returns:
            GeoDataFrame with area column
        """
        if not gdf.empty and 'geometry' in gdf.columns:
            gdf['area_sqm'] = gdf.geometry.area
            gdf['area_ha'] = gdf['area_sqm'] / 10000  # Convert to hectares
        
        return gdf
    
    def _clip_to_parcels(self, detection_gdf: gpd.GeoDataFrame, 
                        parcel_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clip detections to parcel boundaries
        
        Args:
            detection_gdf: Detection GeoDataFrame
            parcel_gdf: Parcel GeoDataFrame
            
        Returns:
            Clipped GeoDataFrame
        """
        try:
            if parcel_gdf.empty or detection_gdf.empty:
                return detection_gdf
            
            # Ensure same CRS
            if parcel_gdf.crs != detection_gdf.crs:
                parcel_gdf = parcel_gdf.to_crs(detection_gdf.crs)
            
            # Clip detections to parcels
            clipped = gpd.clip(detection_gdf, parcel_gdf)
            
            # Recalculate areas after clipping
            if self.calculate_area:
                clipped = self._calculate_areas(clipped)
            
            return clipped
            
        except Exception as e:
            logger.error(f"Error clipping to parcels: {str(e)}")
            return detection_gdf
    
    def _create_statistics(self, detection_gdf: gpd.GeoDataFrame,
                          parcel_gdf: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
        """Create statistics DataFrame
        
        Args:
            detection_gdf: Detection GeoDataFrame
            parcel_gdf: Optional parcel GeoDataFrame
            
        Returns:
            Statistics DataFrame
        """
        stats = []
        
        # Overall statistics
        if not detection_gdf.empty:
            for class_name in detection_gdf['class_name'].unique():
                class_data = detection_gdf[detection_gdf['class_name'] == class_name]
                
                stat = {
                    'class_name': class_name,
                    'count': len(class_data),
                    'total_area_sqm': class_data['area_sqm'].sum() if 'area_sqm' in class_data else 0,
                    'total_area_ha': class_data['area_ha'].sum() if 'area_ha' in class_data else 0,
                    'avg_confidence': class_data['confidence'].mean() if 'confidence' in class_data else 0
                }
                stats.append(stat)
        
        # Per-parcel statistics if available
        if parcel_gdf is not None and not parcel_gdf.empty and not detection_gdf.empty:
            for idx, parcel in parcel_gdf.iterrows():
                if 'geometry' not in parcel or parcel.geometry is None:
                    continue
                
                # Find detections within this parcel
                parcel_detections = detection_gdf[
                    detection_gdf.geometry.within(parcel.geometry)
                ]
                
                if not parcel_detections.empty:
                    for class_name in parcel_detections['class_name'].unique():
                        class_data = parcel_detections[
                            parcel_detections['class_name'] == class_name
                        ]
                        
                        stat = {
                            'parcel_id': parcel.get('id'),
                            'parcel_pnu': parcel.get('pnu'),
                            'class_name': class_name,
                            'count': len(class_data),
                            'area_sqm': class_data['area_sqm'].sum() if 'area_sqm' in class_data else 0,
                            'area_ha': class_data['area_ha'].sum() if 'area_ha' in class_data else 0
                        }
                        stats.append(stat)
        
        if stats:
            return pd.DataFrame(stats)
        else:
            return pd.DataFrame()
    
    def _export_gpkg(self, detection_gdf: gpd.GeoDataFrame,
                    parcel_gdf: Optional[gpd.GeoDataFrame],
                    clipped_gdf: gpd.GeoDataFrame,
                    statistics_df: pd.DataFrame) -> Path:
        """Export to GeoPackage
        
        Args:
            detection_gdf: Detection GeoDataFrame
            parcel_gdf: Parcel GeoDataFrame
            clipped_gdf: Clipped detection GeoDataFrame
            statistics_df: Statistics DataFrame
            
        Returns:
            Path to exported GPKG file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpkg_path = self.output_dir / f"nongview_results_{timestamp}.gpkg"
        
        try:
            # Export detection layer
            if not detection_gdf.empty:
                detection_gdf.to_file(gpkg_path, layer='detections', driver='GPKG')
                self.results['layers']['detections'] = len(detection_gdf)
                logger.info(f"Exported {len(detection_gdf)} detections")
            
            # Export parcel layer
            if parcel_gdf is not None and not parcel_gdf.empty:
                parcel_gdf.to_file(gpkg_path, layer='parcels', driver='GPKG')
                self.results['layers']['parcels'] = len(parcel_gdf)
                logger.info(f"Exported {len(parcel_gdf)} parcels")
            
            # Export clipped layer if different from detections
            if not clipped_gdf.equals(detection_gdf) and not clipped_gdf.empty:
                clipped_gdf.to_file(gpkg_path, layer='clipped_detections', driver='GPKG')
                self.results['layers']['clipped_detections'] = len(clipped_gdf)
                logger.info(f"Exported {len(clipped_gdf)} clipped detections")
            
            # Export statistics as attributes table (without geometry)
            if not statistics_df.empty:
                # Create a dummy GeoDataFrame for statistics
                stats_gdf = gpd.GeoDataFrame(statistics_df)
                stats_gdf.to_file(gpkg_path, layer='statistics', driver='GPKG')
                self.results['layers']['statistics'] = len(statistics_df)
                logger.info(f"Exported {len(statistics_df)} statistics records")
            
            # Add metadata if configured
            if self.include_metadata:
                self._add_metadata_to_gpkg(gpkg_path)
            
            self.results['exported_files'].append(str(gpkg_path))
            logger.info(f"GeoPackage exported: {gpkg_path}")
            
            return gpkg_path
            
        except Exception as e:
            logger.error(f"Error exporting GPKG: {str(e)}")
            raise
    
    def _add_metadata_to_gpkg(self, gpkg_path: Path) -> None:
        """Add metadata to GeoPackage
        
        Args:
            gpkg_path: Path to GPKG file
        """
        try:
            import sqlite3
            
            # Connect to GPKG (SQLite database)
            conn = sqlite3.connect(str(gpkg_path))
            cursor = conn.cursor()
            
            # Create metadata table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            # Add metadata
            metadata_items = [
                ('created_at', datetime.now().isoformat()),
                ('created_by', 'Nong-View2'),
                ('version', '1.0.0'),
                ('crs', self.coordinate_system),
                ('processing_date', datetime.now().strftime("%Y-%m-%d")),
                ('pod_pipeline', 'POD1-6 Complete')
            ]
            
            for key, value in metadata_items:
                cursor.execute(
                    'INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)',
                    (key, value)
                )
            
            conn.commit()
            conn.close()
            
            logger.info("Metadata added to GPKG")
            
        except Exception as e:
            logger.error(f"Error adding metadata: {str(e)}")
    
    def _generate_visualizations(self, detection_gdf: gpd.GeoDataFrame,
                                parcel_gdf: Optional[gpd.GeoDataFrame]) -> None:
        """Generate visualization screenshots
        
        Args:
            detection_gdf: Detection GeoDataFrame
            parcel_gdf: Parcel GeoDataFrame
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Plot parcels if available
            if parcel_gdf is not None and not parcel_gdf.empty and 'geometry' in parcel_gdf:
                parcel_gdf.plot(ax=ax, color='lightgray', edgecolor='black', 
                               alpha=0.3, linewidth=0.5)
            
            # Plot detections by class
            if not detection_gdf.empty and 'geometry' in detection_gdf:
                # Define colors for each class
                class_colors = {
                    '생육기_사료작물': 'green',
                    '생산기_사료작물': 'darkgreen',
                    '곤포_사일리지': 'brown',
                    '비닐하우스_단동': 'blue',
                    '비닐하우스_연동': 'darkblue',
                    '경작지_드론': 'orange',
                    '경작지_위성': 'red'
                }
                
                # Plot each class
                for class_name, color in class_colors.items():
                    class_data = detection_gdf[detection_gdf['class_name'] == class_name]
                    if not class_data.empty:
                        class_data.plot(ax=ax, color=color, alpha=0.6, 
                                      edgecolor='black', linewidth=0.5)
                
                # Create legend
                legend_elements = [
                    Patch(facecolor=color, label=class_name)
                    for class_name, color in class_colors.items()
                    if class_name in detection_gdf['class_name'].values
                ]
                ax.legend(handles=legend_elements, loc='upper right')
            
            ax.set_title('Nong-View2 Analysis Results', fontsize=14, fontweight='bold')
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')
            ax.grid(True, alpha=0.3)
            
            # Save figure
            viz_path = self.output_dir / "visualization.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.results['visualization_path'] = str(viz_path)
            logger.info(f"Visualization saved: {viz_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
    
    def _generate_report(self, statistics_df: pd.DataFrame) -> Path:
        """Generate analysis report
        
        Args:
            statistics_df: Statistics DataFrame
            
        Returns:
            Path to report file
        """
        try:
            report_path = self.output_dir / "analysis_report.html"
            
            html_content = f"""
            <html>
            <head>
                <title>Nong-View2 Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2e7d32; }}
                    h2 {{ color: #1565c0; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metadata {{ background-color: #f5f5f5; padding: 10px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Nong-View2 농업 분석 보고서</h1>
                
                <div class="metadata">
                    <h2>메타데이터</h2>
                    <p><strong>생성일시:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p><strong>좌표계:</strong> {self.coordinate_system}</p>
                    <p><strong>처리 파이프라인:</strong> POD1-6 Complete</p>
                </div>
                
                <h2>분석 통계</h2>
            """
            
            if not statistics_df.empty:
                # Overall statistics
                overall_stats = statistics_df[statistics_df.columns[
                    ~statistics_df.columns.str.contains('parcel')
                ]].drop_duplicates()
                
                if not overall_stats.empty:
                    html_content += "<h3>전체 통계</h3>"
                    html_content += overall_stats.to_html(index=False)
                
                # Per-parcel statistics
                parcel_stats = statistics_df[statistics_df.columns[
                    statistics_df.columns.str.contains('parcel') | 
                    statistics_df.columns.isin(['class_name', 'count', 'area_ha'])
                ]].dropna()
                
                if not parcel_stats.empty:
                    html_content += "<h3>필지별 통계</h3>"
                    html_content += parcel_stats.to_html(index=False)
            
            html_content += """
            </body>
            </html>
            """
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return self.output_dir / "report_error.txt"
    
    def _save_results(self) -> None:
        """Save export results"""
        try:
            results_path = self.output_dir / "export_results.json"
            self.results['metadata']['updated_at'] = datetime.now().isoformat()
            
            FileUtils.save_json(self.results, results_path)
            logger.info(f"Results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")