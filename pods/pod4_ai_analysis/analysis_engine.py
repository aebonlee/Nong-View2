"""
POD4: AI Analysis engine for YOLOv11 inference
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import numpy as np
import cv2
import torch
from PIL import Image

from ...core.config import get_config
from ...core.logger import get_logger
from ...core.utils import FileUtils
from .model_manager import ModelManager

logger = get_logger(__name__)


class AnalysisEngine:
    """Main engine for POD4 AI analysis operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize analysis engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_config().get('ai_analysis')
        
        # Setup directories
        self.input_dir = get_config().output_dir / "pod3_output"
        self.output_dir = get_config().output_dir / "pod4_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis parameters
        self.model_name = self.config.get('model_name', 'yolov11x-seg')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.25)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = self.config.get('batch_size', 8)
        self.save_visualizations = self.config.get('save_visualizations', True)
        self.quality_filter = self.config.get('quality_filter', True)
        
        # Class definitions
        self.classes = self.config.get('classes', {
            0: "생육기_사료작물",
            1: "생산기_사료작물",
            2: "곤포_사일리지",
            3: "비닐하우스_단동",
            4: "비닐하우스_연동",
            5: "경작지_드론",
            6: "경작지_위성"
        })
        
        # Initialize model manager
        self.model_manager = ModelManager(
            model_name=self.model_name,
            device=self.device,
            classes=self.classes
        )
        
        # Initialize results
        self.results = {
            'detections': [],
            'statistics': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'model': self.model_name,
                'device': self.device,
                'classes': self.classes
            }
        }
        
        logger.info(f"POD4 Analysis Engine initialized (model={self.model_name}, device={self.device})")
    
    def process(self, tile_source: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Main processing method for AI analysis
        
        Args:
            tile_source: Path to tiles or directory (uses POD3 output if None)
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting POD4 AI analysis process")
        
        try:
            # Load model
            if not self.model_manager.load_model():
                raise Exception("Failed to load YOLO model")
            
            # Get tiles to process
            tiles = self._get_tiles_to_process(tile_source)
            logger.info(f"Found {len(tiles)} tiles to analyze")
            
            # Process tiles in batches
            total_tiles = len(tiles)
            for i in range(0, total_tiles, self.batch_size):
                batch = tiles[i:min(i + self.batch_size, total_tiles)]
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_tiles + self.batch_size - 1)//self.batch_size}")
                self._process_batch(batch)
            
            # Apply quality filters if configured
            if self.quality_filter:
                self._apply_quality_filters()
            
            # Calculate statistics
            self._calculate_statistics()
            
            # Save results
            self._save_results()
            
            self.results['status'] = 'completed'
            self.results['output_path'] = str(self.output_dir)
            
            logger.info(f"POD4 processing completed. Output: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error in POD4 processing: {str(e)}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
        finally:
            # Clean up model
            self.model_manager.cleanup()
        
        return self.results
    
    def _get_tiles_to_process(self, tile_source: Optional[Union[str, Path]]) -> List[Dict]:
        """Get list of tiles to process
        
        Args:
            tile_source: Tile source path
            
        Returns:
            List of tile information dictionaries
        """
        tiles = []
        
        if tile_source is None:
            # Use POD3 output
            results_file = self.input_dir / "tiling_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    pod3_results = json.load(f)
                    tiles = pod3_results.get('tiles', [])
            else:
                # Fall back to searching for tile files
                tiles_dir = self.input_dir / "tiles"
                if tiles_dir.exists():
                    for tile_path in tiles_dir.rglob("*.tif"):
                        tiles.append({
                            'tile_path': str(tile_path),
                            'id': tile_path.stem
                        })
        else:
            # Use provided source
            tile_source = Path(tile_source)
            if tile_source.is_file():
                tiles.append({
                    'tile_path': str(tile_source),
                    'id': tile_source.stem
                })
            else:
                for tile_path in tile_source.rglob("*.tif"):
                    tiles.append({
                        'tile_path': str(tile_path),
                        'id': tile_path.stem
                    })
        
        return tiles
    
    def _process_batch(self, batch: List[Dict]) -> None:
        """Process a batch of tiles
        
        Args:
            batch: List of tile information dictionaries
        """
        try:
            # Prepare batch images
            batch_images = []
            valid_tiles = []
            
            for tile_info in batch:
                tile_path = Path(tile_info['tile_path'])
                if not tile_path.exists():
                    logger.warning(f"Tile not found: {tile_path}")
                    continue
                
                # Read image
                img = cv2.imread(str(tile_path))
                if img is None:
                    logger.warning(f"Failed to read tile: {tile_path}")
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                batch_images.append(img)
                valid_tiles.append(tile_info)
            
            if not batch_images:
                return
            
            # Run inference
            predictions = self.model_manager.predict(
                batch_images,
                conf=self.confidence_threshold,
                iou=self.iou_threshold
            )
            
            # Process predictions
            for tile_info, pred, img in zip(valid_tiles, predictions, batch_images):
                self._process_prediction(tile_info, pred, img)
                
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
    
    def _process_prediction(self, tile_info: Dict, prediction: Any, image: np.ndarray) -> None:
        """Process prediction for a single tile
        
        Args:
            tile_info: Tile information
            prediction: YOLO prediction result
            image: Original image array
        """
        try:
            tile_path = Path(tile_info['tile_path'])
            
            # Extract detections
            detections = []
            
            if prediction is not None and len(prediction) > 0:
                # Get boxes, masks, and other info
                boxes = prediction.boxes if hasattr(prediction, 'boxes') else None
                masks = prediction.masks if hasattr(prediction, 'masks') else None
                
                if boxes is not None:
                    for i in range(len(boxes)):
                        detection = {
                            'tile_id': tile_info['id'],
                            'tile_path': str(tile_path),
                            'bbox': boxes.xyxy[i].cpu().numpy().tolist() if hasattr(boxes, 'xyxy') else None,
                            'confidence': float(boxes.conf[i]) if hasattr(boxes, 'conf') else 0.0,
                            'class_id': int(boxes.cls[i]) if hasattr(boxes, 'cls') else 0,
                            'class_name': self.classes.get(int(boxes.cls[i]), 'unknown') if hasattr(boxes, 'cls') else 'unknown'
                        }
                        
                        # Add mask if available
                        if masks is not None and i < len(masks.data):
                            mask_data = masks.data[i].cpu().numpy()
                            detection['mask'] = self._encode_mask(mask_data)
                            detection['mask_area'] = float(np.sum(mask_data))
                        
                        detections.append(detection)
            
            # Save detections
            if detections:
                # Add to results
                self.results['detections'].extend(detections)
                
                # Save individual tile results
                tile_results_path = self.output_dir / "detections" / f"{tile_info['id']}_detections.json"
                tile_results_path.parent.mkdir(parents=True, exist_ok=True)
                FileUtils.save_json(detections, tile_results_path)
                
                # Save visualization if configured
                if self.save_visualizations:
                    self._save_visualization(tile_info, prediction, image)
            
            logger.debug(f"Processed tile {tile_info['id']}: {len(detections)} detections")
            
        except Exception as e:
            logger.error(f"Error processing prediction for tile {tile_info['id']}: {str(e)}")
    
    def _encode_mask(self, mask: np.ndarray) -> str:
        """Encode mask as RLE (Run-Length Encoding) string
        
        Args:
            mask: Binary mask array
            
        Returns:
            RLE encoded string
        """
        # Simple RLE encoding
        mask_flat = mask.flatten()
        changes = np.where(mask_flat[:-1] != mask_flat[1:])[0] + 1
        changes = np.concatenate(([0], changes, [len(mask_flat)]))
        runs = np.diff(changes)
        
        # Convert to string
        rle = ' '.join(str(r) for r in runs)
        return rle
    
    def _save_visualization(self, tile_info: Dict, prediction: Any, image: np.ndarray) -> None:
        """Save visualization of predictions
        
        Args:
            tile_info: Tile information
            prediction: YOLO prediction
            image: Original image
        """
        try:
            # Create visualization directory
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Draw predictions on image
            if prediction is not None and hasattr(prediction, 'plot'):
                # Use YOLO's built-in plot function
                annotated = prediction.plot(img=image)
            else:
                # Manual drawing
                annotated = image.copy()
                if prediction and hasattr(prediction, 'boxes'):
                    boxes = prediction.boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        cls_id = int(boxes.cls[i])
                        conf = float(boxes.conf[i])
                        
                        # Draw box
                        color = self._get_color_for_class(cls_id)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{self.classes.get(cls_id, 'unknown')} {conf:.2f}"
                        cv2.putText(annotated, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save visualization
            viz_path = viz_dir / f"{tile_info['id']}_viz.jpg"
            cv2.imwrite(str(viz_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class visualization
        
        Args:
            class_id: Class ID
            
        Returns:
            RGB color tuple
        """
        colors = [
            (255, 0, 0),      # Red - 생육기_사료작물
            (0, 255, 0),      # Green - 생산기_사료작물
            (0, 0, 255),      # Blue - 곤포_사일리지
            (255, 255, 0),    # Yellow - 비닐하우스_단동
            (255, 0, 255),    # Magenta - 비닐하우스_연동
            (0, 255, 255),    # Cyan - 경작지_드론
            (128, 0, 128),    # Purple - 경작지_위성
        ]
        return colors[class_id % len(colors)]
    
    def _apply_quality_filters(self) -> None:
        """Apply quality filters to detections"""
        try:
            filtered_detections = []
            
            for detection in self.results['detections']:
                # Filter by confidence
                if detection['confidence'] < self.confidence_threshold:
                    continue
                
                # Filter by bbox size (remove too small detections)
                if detection.get('bbox'):
                    x1, y1, x2, y2 = detection['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Skip very small detections
                    if width < 10 or height < 10:
                        continue
                    
                    # Skip detections with extreme aspect ratios
                    aspect_ratio = width / height if height > 0 else 0
                    if aspect_ratio > 10 or aspect_ratio < 0.1:
                        continue
                
                filtered_detections.append(detection)
            
            logger.info(f"Quality filter: {len(self.results['detections'])} -> {len(filtered_detections)} detections")
            self.results['detections'] = filtered_detections
            
        except Exception as e:
            logger.error(f"Error applying quality filters: {str(e)}")
    
    def _calculate_statistics(self) -> None:
        """Calculate analysis statistics"""
        try:
            detections = self.results['detections']
            
            # Count detections per class
            class_counts = {}
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Calculate confidence statistics per class
            class_confidences = {}
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                if class_name not in class_confidences:
                    class_confidences[class_name] = []
                class_confidences[class_name].append(detection['confidence'])
            
            confidence_stats = {}
            for class_name, confidences in class_confidences.items():
                if confidences:
                    confidence_stats[class_name] = {
                        'mean': float(np.mean(confidences)),
                        'std': float(np.std(confidences)),
                        'min': float(np.min(confidences)),
                        'max': float(np.max(confidences))
                    }
            
            self.results['statistics'] = {
                'total_detections': len(detections),
                'class_counts': class_counts,
                'confidence_stats': confidence_stats,
                'unique_tiles': len(set(d['tile_id'] for d in detections)),
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info(f"Statistics: Total detections={len(detections)}, Classes={class_counts}")
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
    
    def _save_results(self) -> None:
        """Save analysis results"""
        try:
            # Save full results
            results_path = self.output_dir / "analysis_results.json"
            self.results['metadata']['updated_at'] = datetime.now().isoformat()
            FileUtils.save_json(self.results, results_path)
            logger.info(f"Results saved to: {results_path}")
            
            # Save summary
            summary = {
                'metadata': self.results['metadata'],
                'statistics': self.results['statistics'],
                'detection_count': len(self.results['detections'])
            }
            
            summary_path = self.output_dir / "analysis_summary.json"
            FileUtils.save_json(summary, summary_path)
            logger.info(f"Summary saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")