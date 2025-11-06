"""
Model manager for YOLOv11 models
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import torch
import numpy as np
from ultralytics import YOLO

from ...core.config import get_config
from ...core.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Manage YOLOv11 models for inference"""
    
    def __init__(self, model_name: str = 'yolov11x-seg', 
                 device: str = 'cpu',
                 classes: Dict[int, str] = None):
        """Initialize model manager
        
        Args:
            model_name: Name of YOLO model
            device: Device to use (cpu/cuda)
            classes: Class definitions
        """
        self.model_name = model_name
        self.device = device
        self.classes = classes or {}
        self.model = None
        
        # Model paths
        self.model_dir = get_config().model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        logger.info(f"Model Manager initialized (model={model_name}, device={self.device})")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load YOLO model
        
        Args:
            model_path: Optional custom model path
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_path:
                # Load custom model
                model_file = Path(model_path)
                if not model_file.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return False
            else:
                # Try to find model in model directory
                model_file = self.model_dir / f"{self.model_name}.pt"
                
                if not model_file.exists():
                    # Try default YOLO model names
                    default_names = [
                        f"{self.model_name}.pt",
                        "yolov11x-seg.pt",
                        "yolov11l-seg.pt",
                        "yolov11m-seg.pt",
                        "yolov11s-seg.pt",
                        "yolov11n-seg.pt",
                        "best.pt",
                        "last.pt"
                    ]
                    
                    for name in default_names:
                        test_path = self.model_dir / name
                        if test_path.exists():
                            model_file = test_path
                            break
                    
                    if not model_file.exists():
                        # Download model if not found
                        logger.info(f"Model not found locally, downloading {self.model_name}")
                        model_file = self.model_name
            
            # Load model
            logger.info(f"Loading model: {model_file}")
            self.model = YOLO(str(model_file))
            
            # Move to device
            self.model.to(self.device)
            
            # Set model to eval mode
            self.model.model.eval()
            
            # Update classes if model has custom names
            if hasattr(self.model.model, 'names'):
                model_classes = self.model.model.names
                if model_classes and isinstance(model_classes, dict):
                    self.classes.update(model_classes)
                    logger.info(f"Updated classes from model: {self.classes}")
            
            logger.info(f"Model loaded successfully: {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, images: Union[List[np.ndarray], np.ndarray, str, List[str]],
                conf: float = 0.25,
                iou: float = 0.45,
                imgsz: int = 640,
                **kwargs) -> List:
        """Run inference on images
        
        Args:
            images: Input images (arrays or paths)
            conf: Confidence threshold
            iou: IOU threshold for NMS
            imgsz: Image size for inference
            **kwargs: Additional YOLO predict arguments
            
        Returns:
            List of predictions
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Run inference
            results = self.model.predict(
                source=images,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
                **kwargs
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return []
    
    def predict_batch(self, image_paths: List[str],
                     batch_size: int = 8,
                     conf: float = 0.25,
                     iou: float = 0.45) -> Dict[str, Any]:
        """Run batch inference on multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            Dictionary of results indexed by image path
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        results_dict = {}
        
        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                # Run inference on batch
                batch_results = self.predict(
                    batch_paths,
                    conf=conf,
                    iou=iou
                )
                
                # Map results to paths
                for path, result in zip(batch_paths, batch_results):
                    results_dict[path] = self._parse_result(result)
            
            return results_dict
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            return {}
    
    def _parse_result(self, result) -> Dict[str, Any]:
        """Parse YOLO result into structured format
        
        Args:
            result: YOLO result object
            
        Returns:
            Parsed result dictionary
        """
        parsed = {
            'boxes': [],
            'masks': [],
            'keypoints': [],
            'probs': None
        }
        
        try:
            # Extract boxes
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box_dict = {
                        'xyxy': boxes.xyxy[i].cpu().numpy().tolist() if hasattr(boxes, 'xyxy') else None,
                        'xywh': boxes.xywh[i].cpu().numpy().tolist() if hasattr(boxes, 'xywh') else None,
                        'conf': float(boxes.conf[i]) if hasattr(boxes, 'conf') else 0.0,
                        'cls': int(boxes.cls[i]) if hasattr(boxes, 'cls') else 0,
                        'cls_name': self.classes.get(int(boxes.cls[i]), 'unknown') if hasattr(boxes, 'cls') else 'unknown'
                    }
                    parsed['boxes'].append(box_dict)
            
            # Extract masks (for segmentation models)
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks
                for i in range(len(masks.data)):
                    mask_array = masks.data[i].cpu().numpy()
                    parsed['masks'].append({
                        'mask': mask_array,
                        'segments': masks.xy[i].tolist() if hasattr(masks, 'xy') and i < len(masks.xy) else None
                    })
            
            # Extract keypoints (for pose models)
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints
                for i in range(len(keypoints.data)):
                    parsed['keypoints'].append({
                        'xy': keypoints.xy[i].cpu().numpy().tolist() if hasattr(keypoints, 'xy') else None,
                        'conf': keypoints.conf[i].cpu().numpy().tolist() if hasattr(keypoints, 'conf') else None
                    })
            
            # Extract class probabilities (for classification models)
            if hasattr(result, 'probs') and result.probs is not None:
                parsed['probs'] = result.probs.data.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error parsing result: {str(e)}")
        
        return parsed
    
    def train(self, data_yaml: str, epochs: int = 100, 
              imgsz: int = 640, **kwargs) -> bool:
        """Train YOLO model
        
        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of training epochs
            imgsz: Image size for training
            **kwargs: Additional training arguments
            
        Returns:
            True if training successful
        """
        if self.model is None:
            logger.error("Model not loaded")
            return False
        
        try:
            logger.info(f"Starting training with data: {data_yaml}")
            
            # Train model
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                device=self.device,
                project=str(self.model_dir / 'runs'),
                name='train',
                **kwargs
            )
            
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
    
    def export(self, format: str = 'onnx', **kwargs) -> Optional[str]:
        """Export model to different format
        
        Args:
            format: Export format (onnx, torchscript, etc.)
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported model or None
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            logger.info(f"Exporting model to {format}")
            
            # Export model
            path = self.model.export(
                format=format,
                **kwargs
            )
            
            logger.info(f"Model exported to: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            return None
    
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self.model is not None:
                # Clear CUDA cache if using GPU
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Delete model reference
                del self.model
                self.model = None
                
                logger.info("Model manager cleaned up")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.model is not None
        }
        
        if self.model is not None:
            try:
                info.update({
                    'type': type(self.model).__name__,
                    'task': getattr(self.model, 'task', 'unknown'),
                    'classes': self.classes,
                    'num_classes': len(self.classes)
                })
                
                # Add model architecture info if available
                if hasattr(self.model, 'model'):
                    model = self.model.model
                    info['parameters'] = sum(p.numel() for p in model.parameters())
                    info['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
            except Exception as e:
                logger.error(f"Error getting model info: {str(e)}")
        
        return info