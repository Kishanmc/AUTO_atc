"""
Animal Detection Module
Detects cattle and buffaloes in images using YOLO-based models
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AnimalDetector:
    """Detects animals (cattle/buffalo) in images using YOLO model."""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the animal detector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {
            0: 'cattle',
            1: 'buffalo'
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Animal detection model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load animal detection model: {e}")
            raise
    
    def detect_animals(self, image: np.ndarray) -> Dict:
        """
        Detect animals in the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Run inference
            results = self.model(image, conf=self.confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, 'unknown')
                        }
                        detections.append(detection)
            
            # Find the best detection (highest confidence)
            best_detection = None
            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])
            
            return {
                'detected': len(detections) > 0,
                'detections': detections,
                'best_detection': best_detection,
                'confidence': best_detection['confidence'] if best_detection else 0.0,
                'bounding_box': best_detection['bbox'] if best_detection else None,
                'animal_type': best_detection['class_name'] if best_detection else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error in animal detection: {e}")
            return {
                'detected': False,
                'detections': [],
                'best_detection': None,
                'confidence': 0.0,
                'bounding_box': None,
                'animal_type': 'unknown',
                'error': str(e)
            }
    
    def crop_animal(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crop the animal region from the image.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped image
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            cropped = image[y1:y2, x1:x2]
            return cropped
            
        except Exception as e:
            logger.error(f"Error cropping animal: {e}")
            return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize if too large
            h, w = image.shape[:2]
            max_size = 1024
            
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def get_detection_visualization(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Create visualization of detections on the image.
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Image with detection boxes drawn
        """
        try:
            vis_image = image.copy()
            
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error creating detection visualization: {e}")
            return image
