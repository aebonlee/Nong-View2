"""
POD5: Merging engine for combining AI analysis results
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon, mapping, shape
from shapely.ops import unary_union
from rtree import index
import rasterio
from rasterio.transform import from_bounds

from ...core.config import get_config
from ...core.logger import get_logger
from ...core.utils import FileUtils

logger = get_logger(__name__)


class MergingEngine:
    """Main engine for POD5 merging operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize merging engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_config().get('merging')
        
        # Setup directories
        self.input_dir = get_config().output_dir / "pod4_output"
        self.output_dir = get_config().output_dir / "pod5_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Merging parameters
        self.merge_strategy = self.config.get('merge_strategy', 'nms')  # nms, union, overlap
        self.iou_threshold = self.config.get('iou_threshold', 0.5)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.group_by_class = self.config.get('group_by_class', True)
        self.spatial_index_enabled = self.config.get('spatial_index', True)
        
        # Initialize results
        self.results = {
            'merged_detections': [],
            'class_groups': {},
            'statistics': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'merge_strategy': self.merge_strategy,
                'iou_threshold': self.iou_threshold
            }
        }
        
        logger.info(f"POD5 Merging Engine initialized (strategy={self.merge_strategy})")
    
    def process(self, detection_source: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Main processing method for merging
        
        Args:
            detection_source: Path to detections (uses POD4 output if None)
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting POD5 merging process")
        
        try:
            # Load detections
            detections = self._load_detections(detection_source)
            logger.info(f"Loaded {len(detections)} detections")
            
            # Load tile information for coordinate mapping
            tile_info = self._load_tile_info()
            
            # Map detections to global coordinates
            mapped_detections = self._map_to_global_coordinates(detections, tile_info)
            
            # Group detections by class if configured
            if self.group_by_class:
                class_groups = self._group_by_class(mapped_detections)
            else:
                class_groups = {'all': mapped_detections}
            
            # Merge detections for each class
            for class_name, class_detections in class_groups.items():
                logger.info(f"Merging {len(class_detections)} detections for class: {class_name}")
                merged = self._merge_detections(class_detections, class_name)
                self.results['merged_detections'].extend(merged)
                self.results['class_groups'][class_name] = len(merged)
            
            # Post-process merged detections
            self._post_process_detections()
            
            # Calculate statistics
            self._calculate_statistics()
            
            # Save results
            self._save_results()
            
            self.results['status'] = 'completed'
            self.results['output_path'] = str(self.output_dir)
            
            logger.info(f"POD5 processing completed. Output: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error in POD5 processing: {str(e)}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
        
        return self.results
    
    def _load_detections(self, source: Optional[Union[str, Path]]) -> List[Dict]:
        """Load detections from POD4 output
        
        Args:
            source: Detection source path
            
        Returns:
            List of detections
        """
        detections = []
        
        if source is None:
            # Use POD4 output
            results_file = self.input_dir / "analysis_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    pod4_results = json.load(f)
                    detections = pod4_results.get('detections', [])
            else:
                # Load from individual detection files
                detections_dir = self.input_dir / "detections"
                if detections_dir.exists():
                    for det_file in detections_dir.glob("*_detections.json"):
                        with open(det_file, 'r') as f:
                            file_detections = json.load(f)
                            if isinstance(file_detections, list):
                                detections.extend(file_detections)
        else:
            # Load from provided source
            source = Path(source)
            if source.is_file():
                with open(source, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'detections' in data:
                        detections = data['detections']
                    elif isinstance(data, list):
                        detections = data
            else:
                # Load all detection files in directory
                for det_file in source.glob("*.json"):
                    with open(det_file, 'r') as f:
                        file_detections = json.load(f)
                        if isinstance(file_detections, list):
                            detections.extend(file_detections)
        
        return detections
    
    def _load_tile_info(self) -> Dict[str, Any]:
        """Load tile information from POD3
        
        Returns:
            Dictionary of tile information
        """
        tile_info = {}
        
        # Try to load POD3 tiling results
        pod3_dir = get_config().output_dir / "pod3_output"
        tiling_results = pod3_dir / "tiling_results.json"
        
        if tiling_results.exists():
            with open(tiling_results, 'r') as f:
                pod3_results = json.load(f)
                tiles = pod3_results.get('tiles', [])
                
                # Create lookup by tile ID
                for tile in tiles:
                    tile_id = tile.get('id')
                    if tile_id:
                        tile_info[tile_id] = tile
        
        return tile_info
    
    def _map_to_global_coordinates(self, detections: List[Dict], 
                                  tile_info: Dict[str, Any]) -> List[Dict]:
        """Map tile-local detections to global coordinates
        
        Args:
            detections: List of detections
            tile_info: Tile information dictionary
            
        Returns:
            List of detections with global coordinates
        """
        mapped_detections = []
        
        for detection in detections:
            try:
                tile_id = detection.get('tile_id')
                
                # Get tile information
                if tile_id and tile_id in tile_info:
                    tile = tile_info[tile_id]
                    x_offset = tile.get('x_offset', 0)
                    y_offset = tile.get('y_offset', 0)
                    
                    # Map bbox to global coordinates
                    if detection.get('bbox'):
                        local_bbox = detection['bbox']
                        global_bbox = [
                            local_bbox[0] + x_offset,
                            local_bbox[1] + y_offset,
                            local_bbox[2] + x_offset,
                            local_bbox[3] + y_offset
                        ]
                        detection['global_bbox'] = global_bbox
                        
                        # Create geometry
                        detection['geometry'] = box(*global_bbox)
                    
                    # Add tile reference
                    detection['tile_info'] = {
                        'id': tile_id,
                        'bounds': tile.get('bounds'),
                        'source_image': tile.get('source_image')
                    }
                else:
                    # No tile info, use original bbox
                    if detection.get('bbox'):
                        detection['global_bbox'] = detection['bbox']
                        detection['geometry'] = box(*detection['bbox'])
                
                # Filter by confidence
                if detection.get('confidence', 0) >= self.min_confidence:
                    mapped_detections.append(detection)
                    
            except Exception as e:
                logger.error(f"Error mapping detection: {str(e)}")
        
        return mapped_detections
    
    def _group_by_class(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        """Group detections by class
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary of detections grouped by class
        """
        class_groups = {}
        
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            
            if class_name not in class_groups:
                class_groups[class_name] = []
            
            class_groups[class_name].append(detection)
        
        return class_groups
    
    def _merge_detections(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """Merge overlapping detections
        
        Args:
            detections: List of detections to merge
            class_name: Class name for these detections
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
        
        if self.merge_strategy == 'nms':
            return self._nms_merge(detections, class_name)
        elif self.merge_strategy == 'union':
            return self._union_merge(detections, class_name)
        elif self.merge_strategy == 'overlap':
            return self._overlap_merge(detections, class_name)
        else:
            logger.warning(f"Unknown merge strategy: {self.merge_strategy}")
            return detections
    
    def _nms_merge(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """Non-Maximum Suppression merge
        
        Args:
            detections: List of detections
            class_name: Class name
            
        Returns:
            List of merged detections after NMS
        """
        if not detections:
            return []
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Build spatial index if enabled
        if self.spatial_index_enabled:
            idx = index.Index()
            for i, det in enumerate(sorted_detections):
                if det.get('geometry'):
                    idx.insert(i, det['geometry'].bounds)
        
        merged = []
        used = set()
        
        for i, detection in enumerate(sorted_detections):
            if i in used:
                continue
            
            if not detection.get('geometry'):
                continue
            
            current_geom = detection['geometry']
            
            # Find overlapping detections
            overlapping = []
            
            if self.spatial_index_enabled:
                # Use spatial index
                candidates = list(idx.intersection(current_geom.bounds))
                for j in candidates:
                    if j > i and j not in used:
                        other_geom = sorted_detections[j]['geometry']
                        iou = self._calculate_iou(current_geom, other_geom)
                        if iou > self.iou_threshold:
                            overlapping.append(j)
                            used.add(j)
            else:
                # Linear search
                for j in range(i + 1, len(sorted_detections)):
                    if j not in used:
                        other_geom = sorted_detections[j].get('geometry')
                        if other_geom:
                            iou = self._calculate_iou(current_geom, other_geom)
                            if iou > self.iou_threshold:
                                overlapping.append(j)
                                used.add(j)
            
            # Create merged detection
            merged_detection = {
                'id': f"merged_{class_name}_{len(merged):04d}",
                'class_name': class_name,
                'class_id': detection.get('class_id'),
                'confidence': detection.get('confidence'),
                'bbox': detection.get('global_bbox'),
                'geometry': current_geom,
                'merged_count': 1 + len(overlapping),
                'source_detections': [detection.get('tile_id')] + 
                                    [sorted_detections[j].get('tile_id') for j in overlapping]
            }
            
            # If masks available, merge them
            if detection.get('mask'):
                merged_detection['has_mask'] = True
            
            merged.append(merged_detection)
        
        return merged
    
    def _union_merge(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """Union merge strategy - combine overlapping geometries
        
        Args:
            detections: List of detections
            class_name: Class name
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
        
        # Collect all geometries
        geometries = [det['geometry'] for det in detections if det.get('geometry')]
        
        if not geometries:
            return []
        
        # Union all overlapping geometries
        merged_geoms = []
        processed = set()
        
        for i, geom in enumerate(geometries):
            if i in processed:
                continue
            
            # Find all overlapping geometries
            union_geom = geom
            overlapping_indices = [i]
            
            for j in range(i + 1, len(geometries)):
                if j not in processed:
                    if union_geom.intersects(geometries[j]):
                        union_geom = union_geom.union(geometries[j])
                        overlapping_indices.append(j)
                        processed.add(j)
            
            # Calculate average confidence
            avg_confidence = np.mean([
                detections[idx].get('confidence', 0) 
                for idx in overlapping_indices
            ])
            
            # Create merged detection
            bounds = union_geom.bounds
            merged_detection = {
                'id': f"merged_{class_name}_{len(merged_geoms):04d}",
                'class_name': class_name,
                'class_id': detections[overlapping_indices[0]].get('class_id'),
                'confidence': float(avg_confidence),
                'bbox': list(bounds),
                'geometry': union_geom,
                'merged_count': len(overlapping_indices),
                'area': union_geom.area
            }
            
            merged_geoms.append(merged_detection)
        
        return merged_geoms
    
    def _overlap_merge(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """Overlap merge strategy - keep best detection per overlap group
        
        Args:
            detections: List of detections
            class_name: Class name
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
        
        # Group overlapping detections
        groups = []
        used = set()
        
        for i, detection in enumerate(detections):
            if i in used:
                continue
            
            if not detection.get('geometry'):
                continue
            
            # Start new group
            group = [detection]
            group_indices = [i]
            used.add(i)
            
            # Find all overlapping detections
            for j in range(i + 1, len(detections)):
                if j not in used:
                    other = detections[j]
                    if other.get('geometry'):
                        # Check if overlaps with any in group
                        overlaps = any(
                            self._calculate_iou(
                                member.get('geometry'),
                                other.get('geometry')
                            ) > self.iou_threshold
                            for member in group
                        )
                        
                        if overlaps:
                            group.append(other)
                            group_indices.append(j)
                            used.add(j)
            
            groups.append(group)
        
        # Select best detection from each group
        merged = []
        for group in groups:
            # Sort by confidence and select best
            best = max(group, key=lambda x: x.get('confidence', 0))
            
            merged_detection = {
                'id': f"merged_{class_name}_{len(merged):04d}",
                'class_name': class_name,
                'class_id': best.get('class_id'),
                'confidence': best.get('confidence'),
                'bbox': best.get('global_bbox'),
                'geometry': best.get('geometry'),
                'merged_count': len(group)
            }
            
            merged.append(merged_detection)
        
        return merged
    
    def _calculate_iou(self, geom1: Polygon, geom2: Polygon) -> float:
        """Calculate Intersection over Union
        
        Args:
            geom1: First geometry
            geom2: Second geometry
            
        Returns:
            IOU value
        """
        try:
            intersection = geom1.intersection(geom2).area
            union = geom1.union(geom2).area
            
            if union == 0:
                return 0
            
            return intersection / union
            
        except Exception as e:
            logger.error(f"Error calculating IOU: {str(e)}")
            return 0
    
    def _post_process_detections(self) -> None:
        """Post-process merged detections"""
        try:
            # Remove duplicate geometries
            unique_detections = []
            seen_geometries = set()
            
            for detection in self.results['merged_detections']:
                if detection.get('geometry'):
                    geom_wkt = detection['geometry'].wkt
                    if geom_wkt not in seen_geometries:
                        seen_geometries.add(geom_wkt)
                        unique_detections.append(detection)
            
            self.results['merged_detections'] = unique_detections
            
            logger.info(f"Post-processing: {len(self.results['merged_detections'])} unique detections")
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
    
    def _calculate_statistics(self) -> None:
        """Calculate merging statistics"""
        try:
            detections = self.results['merged_detections']
            
            # Class statistics
            class_stats = {}
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                if class_name not in class_stats:
                    class_stats[class_name] = {
                        'count': 0,
                        'total_area': 0,
                        'avg_confidence': []
                    }
                
                class_stats[class_name]['count'] += 1
                
                if detection.get('geometry'):
                    class_stats[class_name]['total_area'] += detection['geometry'].area
                
                if detection.get('confidence'):
                    class_stats[class_name]['avg_confidence'].append(detection['confidence'])
            
            # Calculate averages
            for class_name, stats in class_stats.items():
                if stats['avg_confidence']:
                    stats['avg_confidence'] = float(np.mean(stats['avg_confidence']))
                else:
                    stats['avg_confidence'] = 0
            
            # Overall statistics
            total_area = sum(
                detection.get('geometry').area 
                for detection in detections 
                if detection.get('geometry')
            )
            
            self.results['statistics'] = {
                'total_merged_detections': len(detections),
                'class_statistics': class_stats,
                'total_coverage_area': total_area,
                'merge_reduction_ratio': self._calculate_merge_ratio(),
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info(f"Statistics: {self.results['statistics']}")
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
    
    def _calculate_merge_ratio(self) -> float:
        """Calculate merge reduction ratio
        
        Returns:
            Ratio of merged to original detections
        """
        try:
            total_original = sum(
                det.get('merged_count', 1) 
                for det in self.results['merged_detections']
            )
            
            if total_original == 0:
                return 0
            
            return len(self.results['merged_detections']) / total_original
            
        except:
            return 0
    
    def _save_results(self) -> None:
        """Save merging results"""
        try:
            # Convert geometries to JSON-serializable format
            output_detections = []
            for detection in self.results['merged_detections']:
                output_det = detection.copy()
                if 'geometry' in output_det:
                    output_det['geometry'] = mapping(output_det['geometry'])
                output_detections.append(output_det)
            
            # Save full results
            results_data = {
                'merged_detections': output_detections,
                'class_groups': self.results['class_groups'],
                'statistics': self.results['statistics'],
                'metadata': self.results['metadata']
            }
            
            results_path = self.output_dir / "merging_results.json"
            results_data['metadata']['updated_at'] = datetime.now().isoformat()
            FileUtils.save_json(results_data, results_path)
            logger.info(f"Results saved to: {results_path}")
            
            # Save as GeoJSON for visualization
            if output_detections:
                self._save_geojson(output_detections)
            
            # Save summary
            summary_path = self.output_dir / "merging_summary.json"
            summary = {
                'metadata': self.results['metadata'],
                'statistics': self.results['statistics'],
                'class_groups': self.results['class_groups']
            }
            FileUtils.save_json(summary, summary_path)
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _save_geojson(self, detections: List[Dict]) -> None:
        """Save detections as GeoJSON
        
        Args:
            detections: List of detections with geometry
        """
        try:
            features = []
            
            for detection in detections:
                if detection.get('geometry'):
                    feature = {
                        'type': 'Feature',
                        'geometry': detection['geometry'],
                        'properties': {
                            'id': detection.get('id'),
                            'class_name': detection.get('class_name'),
                            'confidence': detection.get('confidence'),
                            'merged_count': detection.get('merged_count', 1)
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
            
            geojson_path = self.output_dir / "merged_detections.geojson"
            FileUtils.save_json(geojson, geojson_path)
            logger.info(f"GeoJSON saved to: {geojson_path}")
            
        except Exception as e:
            logger.error(f"Error saving GeoJSON: {str(e)}")