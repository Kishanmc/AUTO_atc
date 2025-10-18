"""
Keypoint Detection Module
Detects anatomical keypoints on cattle and buffaloes using MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class KeypointDetector:
    """Detects anatomical keypoints on cattle and buffaloes."""
    
    def __init__(self, model_complexity: int = 1, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the keypoint detector.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define cattle/buffalo specific keypoints
        self.keypoint_names = {
            # Head and neck
            0: 'nose',
            1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
            4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
            7: 'left_ear', 8: 'right_ear',
            9: 'mouth_left', 10: 'mouth_right',
            
            # Shoulder and front legs
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',
            
            # Body and spine
            17: 'spine_upper', 18: 'spine_mid', 19: 'spine_lower',
            20: 'chest_center',
            
            # Hind legs and hips
            21: 'left_hip', 22: 'right_hip',
            23: 'left_knee', 24: 'right_knee',
            25: 'left_ankle', 26: 'right_ankle',
            
            # Tail
            27: 'tail_base', 28: 'tail_tip',
            
            # Additional cattle-specific points
            29: 'hump', 30: 'udder_front', 31: 'udder_back'
        }
        
        # Keypoint connections for visualization
        self.keypoint_connections = [
            # Head connections
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (7, 8), (9, 10),
            
            # Shoulder connections
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            
            # Spine connections
            (17, 18), (18, 19), (17, 20), (18, 20), (19, 20),
            
            # Hip connections
            (21, 22), (21, 23), (22, 24), (23, 25), (24, 26),
            
            # Tail connections
            (19, 27), (27, 28),
            
            # Body connections
            (11, 17), (12, 17), (17, 21), (17, 22),
            
            # Cattle-specific connections
            (17, 29), (29, 30), (30, 31)
        ]
        
        logger.info("Keypoint detector initialized")
    
    def detect_keypoints(self, image: np.ndarray, 
                        bbox: Optional[List[float]] = None) -> Dict:
        """
        Detect keypoints on the animal in the image.
        
        Args:
            image: Input image as numpy array
            bbox: Optional bounding box to focus detection
            
        Returns:
            Dictionary containing keypoint detection results
        """
        try:
            # Crop image to bounding box if provided
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = image.shape[:2]
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                cropped_image = image[y1:y2, x1:x2]
                bbox_offset = (x1, y1)
            else:
                cropped_image = image
                bbox_offset = (0, 0)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            
            # Process image with MediaPipe
            results = self.pose.process(rgb_image)
            
            # Extract keypoints
            keypoints = []
            landmarks = None
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                
                # Convert MediaPipe landmarks to our format
                for i, landmark in enumerate(landmarks.landmark):
                    if i < len(self.keypoint_names):
                        # Adjust coordinates for bounding box offset
                        x = landmark.x * cropped_image.shape[1] + bbox_offset[0]
                        y = landmark.y * cropped_image.shape[0] + bbox_offset[1]
                        
                        keypoint = {
                            'id': i,
                            'name': self.keypoint_names.get(i, f'point_{i}'),
                            'x': float(x),
                            'y': float(y),
                            'z': float(landmark.z),
                            'visibility': float(landmark.visibility),
                            'confidence': float(landmark.visibility)
                        }
                        keypoints.append(keypoint)
            
            # Calculate detection confidence
            confidence = self._calculate_detection_confidence(keypoints)
            
            # Detect additional cattle-specific features
            additional_keypoints = self._detect_cattle_features(image, bbox)
            keypoints.extend(additional_keypoints)
            
            return {
                'detected': len(keypoints) > 0,
                'keypoints': keypoints,
                'confidence': confidence,
                'landmarks': landmarks,
                'image_shape': image.shape,
                'total_keypoints': len(keypoints),
                'bbox_used': bbox is not None
            }
            
        except Exception as e:
            logger.error(f"Error in keypoint detection: {e}")
            return {
                'detected': False,
                'keypoints': [],
                'confidence': 0.0,
                'landmarks': None,
                'image_shape': image.shape,
                'error': str(e)
            }
    
    def _calculate_detection_confidence(self, keypoints: List[Dict]) -> float:
        """Calculate overall detection confidence."""
        try:
            if not keypoints:
                return 0.0
            
            # Calculate average confidence of detected keypoints
            confidences = [kp['confidence'] for kp in keypoints if kp['confidence'] > 0]
            
            if not confidences:
                return 0.0
            
            # Weight by number of keypoints detected
            avg_confidence = np.mean(confidences)
            keypoint_ratio = len(confidences) / len(self.keypoint_names)
            
            # Combine confidence and coverage
            overall_confidence = avg_confidence * (0.7 + 0.3 * keypoint_ratio)
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating detection confidence: {e}")
            return 0.0
    
    def _detect_cattle_features(self, image: np.ndarray, 
                               bbox: Optional[List[float]]) -> List[Dict]:
        """Detect cattle-specific features like hump and udder."""
        try:
            additional_keypoints = []
            
            # Detect hump
            hump_keypoint = self._detect_hump(image, bbox)
            if hump_keypoint:
                additional_keypoints.append(hump_keypoint)
            
            # Detect udder
            udder_keypoints = self._detect_udder(image, bbox)
            additional_keypoints.extend(udder_keypoints)
            
            return additional_keypoints
            
        except Exception as e:
            logger.warning(f"Error detecting cattle features: {e}")
            return []
    
    def _detect_hump(self, image: np.ndarray, bbox: Optional[List[float]]) -> Optional[Dict]:
        """Detect hump on cattle."""
        try:
            # Focus on upper body region
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = image.shape[:2]
                
                # Focus on upper third of bounding box
                upper_region = image[y1:y1 + int((y2-y1)*0.4), x1:x2]
            else:
                h, w = image.shape[:2]
                upper_region = image[:int(h*0.4), :]
            
            # Convert to grayscale
            gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
            
            # Detect contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (potential hump)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Adjust coordinates if bbox was used
                    if bbox:
                        cx += bbox[0]
                        cy += bbox[1]
                    
                    return {
                        'id': 29,
                        'name': 'hump',
                        'x': float(cx),
                        'y': float(cy),
                        'z': 0.0,
                        'visibility': 0.8,
                        'confidence': 0.7
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting hump: {e}")
            return None
    
    def _detect_udder(self, image: np.ndarray, bbox: Optional[List[float]]) -> List[Dict]:
        """Detect udder on cattle."""
        try:
            udder_keypoints = []
            
            # Focus on lower body region
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = image.shape[:2]
                
                # Focus on lower third of bounding box
                lower_region = image[y1 + int((y2-y1)*0.6):y2, x1:x2]
                offset_y = y1 + int((y2-y1)*0.6)
            else:
                h, w = image.shape[:2]
                lower_region = image[int(h*0.6):, :]
                offset_y = int(h*0.6)
            
            # Convert to grayscale
            gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
            
            # Detect contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find contours that could be udder
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area > 100:  # Minimum area threshold
                        # Calculate centroid
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Adjust coordinates
                            cx += bbox[0] if bbox else 0
                            cy += offset_y
                            
                            # Determine if front or back udder
                            keypoint_id = 30 if i == 0 else 31
                            keypoint_name = 'udder_front' if i == 0 else 'udder_back'
                            
                            udder_keypoints.append({
                                'id': keypoint_id,
                                'name': keypoint_name,
                                'x': float(cx),
                                'y': float(cy),
                                'z': 0.0,
                                'visibility': 0.7,
                                'confidence': 0.6
                            })
            
            return udder_keypoints
            
        except Exception as e:
            logger.warning(f"Error detecting udder: {e}")
            return []
    
    def visualize_keypoints(self, image: np.ndarray, 
                          keypoints: List[Dict],
                          connections: bool = True) -> np.ndarray:
        """
        Visualize keypoints on the image.
        
        Args:
            image: Input image
            keypoints: List of detected keypoints
            connections: Whether to draw connections between keypoints
            
        Returns:
            Image with keypoints visualized
        """
        try:
            vis_image = image.copy()
            
            # Draw keypoints
            for kp in keypoints:
                x, y = int(kp['x']), int(kp['y'])
                confidence = kp['confidence']
                
                # Color based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw keypoint
                cv2.circle(vis_image, (x, y), 5, color, -1)
                
                # Draw keypoint name
                cv2.putText(vis_image, kp['name'], (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw connections
            if connections:
                for connection in self.keypoint_connections:
                    start_id, end_id = connection
                    
                    # Find keypoints by ID
                    start_kp = next((kp for kp in keypoints if kp['id'] == start_id), None)
                    end_kp = next((kp for kp in keypoints if kp['id'] == end_id), None)
                    
                    if start_kp and end_kp:
                        start_point = (int(start_kp['x']), int(start_kp['y']))
                        end_point = (int(end_kp['x']), int(end_kp['y']))
                        
                        # Draw line
                        cv2.line(vis_image, start_point, end_point, (255, 0, 0), 2)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error visualizing keypoints: {e}")
            return image
    
    def get_keypoint_connections(self) -> List[Tuple[int, int]]:
        """Get keypoint connections for visualization."""
        return self.keypoint_connections
    
    def get_keypoint_names(self) -> Dict[int, str]:
        """Get keypoint names mapping."""
        return self.keypoint_names
    
    def filter_keypoints_by_confidence(self, keypoints: List[Dict], 
                                     min_confidence: float = 0.5) -> List[Dict]:
        """Filter keypoints by minimum confidence threshold."""
        return [kp for kp in keypoints if kp['confidence'] >= min_confidence]
    
    def get_keypoints_by_name(self, keypoints: List[Dict], 
                            name: str) -> Optional[Dict]:
        """Get keypoint by name."""
        return next((kp for kp in keypoints if kp['name'] == name), None)
    
    def calculate_keypoint_distances(self, keypoints: List[Dict]) -> Dict[str, float]:
        """Calculate distances between keypoints."""
        try:
            distances = {}
            
            # Define important distance measurements
            distance_pairs = [
                ('shoulder_width', 'left_shoulder', 'right_shoulder'),
                ('hip_width', 'left_hip', 'right_hip'),
                ('body_length', 'spine_upper', 'spine_lower'),
                ('leg_length_left', 'left_hip', 'left_ankle'),
                ('leg_length_right', 'right_hip', 'right_ankle')
            ]
            
            for distance_name, start_name, end_name in distance_pairs:
                start_kp = self.get_keypoints_by_name(keypoints, start_name)
                end_kp = self.get_keypoints_by_name(keypoints, end_name)
                
                if start_kp and end_kp:
                    distance = np.sqrt(
                        (start_kp['x'] - end_kp['x'])**2 + 
                        (start_kp['y'] - end_kp['y'])**2
                    )
                    distances[distance_name] = float(distance)
            
            return distances
            
        except Exception as e:
            logger.warning(f"Error calculating keypoint distances: {e}")
            return {}
    
    def get_detector_status(self) -> Dict:
        """Get detector status and configuration."""
        return {
            'status': 'operational',
            'model_complexity': self.model_complexity,
            'min_detection_confidence': self.min_detection_confidence,
            'min_tracking_confidence': self.min_tracking_confidence,
            'total_keypoints': len(self.keypoint_names),
            'keypoint_connections': len(self.keypoint_connections),
            'cattle_features': ['hump', 'udder_front', 'udder_back']
        }