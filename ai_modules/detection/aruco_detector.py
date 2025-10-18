"""
ArUco Marker Detection Module
Detects ArUco markers for scale reference in measurements
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ArucoDetector:
    """Detects ArUco markers for scale reference."""
    
    def __init__(self, dictionary_type: int = 6, marker_size: float = 5.0):
        """
        Initialize the ArUco detector.
        
        Args:
            dictionary_type: ArUco dictionary type (0-9)
            marker_size: Physical size of markers in cm
        """
        self.dictionary_type = dictionary_type
        self.marker_size = marker_size
        
        # Initialize ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        
        # Initialize detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # Camera calibration parameters (would be loaded from calibration file)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Load camera calibration if available
        self._load_camera_calibration()
        
        logger.info(f"ArUco detector initialized with dictionary {dictionary_type}")
    
    def _load_camera_calibration(self):
        """Load camera calibration parameters."""
        try:
            calibration_file = Path("camera_calibration.npz")
            if calibration_file.exists():
                data = np.load(calibration_file)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                logger.info("Camera calibration loaded")
            else:
                # Use default calibration (would need to be calibrated for each camera)
                self.camera_matrix = np.array([
                    [800, 0, 320],
                    [0, 800, 240],
                    [0, 0, 1]
                ], dtype=np.float32)
                self.dist_coeffs = np.zeros((4, 1))
                logger.warning("Using default camera calibration")
        except Exception as e:
            logger.warning(f"Error loading camera calibration: {e}")
            self.camera_matrix = None
            self.dist_coeffs = None
    
    def detect_markers(self, image: np.ndarray) -> Dict:
        """
        Detect ArUco markers in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing marker detection results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )
            
            markers = []
            scale_factor = 1.0
            
            if ids is not None and len(ids) > 0:
                # Estimate pose for each marker
                for i, marker_id in enumerate(ids.flatten()):
                    marker_corners = corners[i][0]
                    
                    # Calculate marker center
                    center = np.mean(marker_corners, axis=0)
                    
                    # Calculate marker size in pixels
                    pixel_size = self._calculate_marker_size(marker_corners)
                    
                    # Calculate scale factor
                    if pixel_size > 0:
                        marker_scale_factor = self.marker_size / pixel_size
                        scale_factor = marker_scale_factor
                    
                    # Estimate pose if camera calibration is available
                    pose = None
                    if self.camera_matrix is not None:
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [marker_corners], self.marker_size, 
                            self.camera_matrix, self.dist_coeffs
                        )
                        if rvecs is not None and tvecs is not None:
                            pose = {
                                'rotation': rvecs[0].flatten().tolist(),
                                'translation': tvecs[0].flatten().tolist()
                            }
                    
                    marker_info = {
                        'id': int(marker_id),
                        'corners': marker_corners.tolist(),
                        'center': center.tolist(),
                        'pixel_size': float(pixel_size),
                        'scale_factor': float(scale_factor),
                        'pose': pose
                    }
                    markers.append(marker_info)
            
            return {
                'detected': len(markers) > 0,
                'markers': markers,
                'count': len(markers),
                'scale_factor': scale_factor,
                'dictionary_type': self.dictionary_type,
                'marker_size_cm': self.marker_size
            }
            
        except Exception as e:
            logger.error(f"Error detecting ArUco markers: {e}")
            return {
                'detected': False,
                'markers': [],
                'count': 0,
                'scale_factor': 1.0,
                'error': str(e)
            }
    
    def _calculate_marker_size(self, corners: np.ndarray) -> float:
        """Calculate marker size in pixels."""
        try:
            # Calculate average side length
            side_lengths = []
            
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i + 1) % 4]
                length = np.linalg.norm(p2 - p1)
                side_lengths.append(length)
            
            return np.mean(side_lengths)
            
        except Exception as e:
            logger.warning(f"Error calculating marker size: {e}")
            return 0.0
    
    def calculate_scale_factor(self, markers: List[Dict]) -> float:
        """
        Calculate scale factor from detected markers.
        
        Args:
            markers: List of detected markers
            
        Returns:
            Scale factor (cm per pixel)
        """
        try:
            if not markers:
                return 1.0
            
            # Use the marker with the highest confidence (largest size)
            best_marker = max(markers, key=lambda m: m['pixel_size'])
            
            if best_marker['pixel_size'] > 0:
                scale_factor = self.marker_size / best_marker['pixel_size']
                return float(scale_factor)
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating scale factor: {e}")
            return 1.0
    
    def visualize_markers(self, image: np.ndarray, 
                         markers: List[Dict],
                         draw_axes: bool = True) -> np.ndarray:
        """
        Visualize detected markers on the image.
        
        Args:
            image: Input image
            markers: List of detected markers
            draw_axes: Whether to draw pose axes
            
        Returns:
            Image with markers visualized
        """
        try:
            vis_image = image.copy()
            
            for marker in markers:
                corners = np.array(marker['corners'], dtype=np.int32)
                marker_id = marker['id']
                center = tuple(map(int, marker['center']))
                
                # Draw marker outline
                cv2.polylines(vis_image, [corners], True, (0, 255, 0), 2)
                
                # Draw marker ID
                cv2.putText(vis_image, f"ID: {marker_id}", 
                           (center[0] - 20, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw scale factor
                scale_factor = marker['scale_factor']
                cv2.putText(vis_image, f"Scale: {scale_factor:.3f} cm/px",
                           (center[0] - 20, center[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw pose axes if available
                if draw_axes and marker['pose'] and self.camera_matrix is not None:
                    rvec = np.array(marker['pose']['rotation'])
                    tvec = np.array(marker['pose']['translation'])
                    
                    cv2.aruco.drawAxis(vis_image, self.camera_matrix, 
                                     self.dist_coeffs, rvec, tvec, 
                                     self.marker_size * 0.5)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error visualizing markers: {e}")
            return image
    
    def generate_marker_image(self, marker_id: int, 
                            size_pixels: int = 200) -> np.ndarray:
        """
        Generate an ArUco marker image.
        
        Args:
            marker_id: ID of the marker to generate
            size_pixels: Size of the generated marker in pixels
            
        Returns:
            Generated marker image
        """
        try:
            marker_image = cv2.aruco.generateImageMarker(
                self.aruco_dict, marker_id, size_pixels
            )
            return marker_image
            
        except Exception as e:
            logger.error(f"Error generating marker image: {e}")
            return np.zeros((size_pixels, size_pixels), dtype=np.uint8)
    
    def save_marker_image(self, marker_id: int, 
                         output_path: str,
                         size_pixels: int = 200) -> bool:
        """
        Save an ArUco marker image to file.
        
        Args:
            marker_id: ID of the marker to generate
            output_path: Path to save the marker
            size_pixels: Size of the generated marker in pixels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            marker_image = self.generate_marker_image(marker_id, size_pixels)
            cv2.imwrite(output_path, marker_image)
            logger.info(f"Marker {marker_id} saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving marker image: {e}")
            return False
    
    def calibrate_camera(self, calibration_images: List[np.ndarray],
                        marker_size: float = 5.0) -> Dict:
        """
        Calibrate camera using ArUco markers.
        
        Args:
            calibration_images: List of calibration images
            marker_size: Physical size of markers in cm
            
        Returns:
            Calibration results
        """
        try:
            all_corners = []
            all_ids = []
            
            # Detect markers in all calibration images
            for image in calibration_images:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.detector_params
                )
                
                if ids is not None and len(ids) > 0:
                    all_corners.append(corners)
                    all_ids.append(ids)
            
            if not all_corners:
                return {'success': False, 'error': 'No markers detected in calibration images'}
            
            # Prepare object points
            obj_points = []
            img_points = []
            
            for corners in all_corners:
                for corner in corners:
                    # Create object points for this marker
                    obj_point = np.array([
                        [-marker_size/2, marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)
                    
                    obj_points.append(obj_point)
                    img_points.append(corner.reshape(-1, 2))
            
            # Calibrate camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, 
                calibration_images[0].shape[:2][::-1], 
                None, None
            )
            
            if ret:
                # Save calibration
                np.savez('camera_calibration.npz',
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs)
                
                # Update instance variables
                self.camera_matrix = camera_matrix
                self.dist_coeffs = dist_coeffs
                
                return {
                    'success': True,
                    'camera_matrix': camera_matrix.tolist(),
                    'dist_coeffs': dist_coeffs.tolist(),
                    'reprojection_error': ret
                }
            else:
                return {'success': False, 'error': 'Camera calibration failed'}
                
        except Exception as e:
            logger.error(f"Error calibrating camera: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_detector_status(self) -> Dict:
        """Get detector status and configuration."""
        return {
            'status': 'operational',
            'dictionary_type': self.dictionary_type,
            'marker_size_cm': self.marker_size,
            'camera_calibrated': self.camera_matrix is not None,
            'detector_params': {
                'adaptiveThreshWinSizeMin': self.detector_params.adaptiveThreshWinSizeMin,
                'adaptiveThreshWinSizeMax': self.detector_params.adaptiveThreshWinSizeMax,
                'adaptiveThreshWinSizeStep': self.detector_params.adaptiveThreshWinSizeStep,
                'adaptiveThreshConstant': self.detector_params.adaptiveThreshConstant,
                'minMarkerPerimeterRate': self.detector_params.minMarkerPerimeterRate,
                'maxMarkerPerimeterRate': self.detector_params.maxMarkerPerimeterRate,
                'polygonalApproxAccuracyRate': self.detector_params.polygonalApproxAccuracyRate,
                'minCornerDistanceRate': self.detector_params.minCornerDistanceRate,
                'minDistanceToBorder': self.detector_params.minDistanceToBorder,
                'minMarkerDistanceRate': self.detector_params.minMarkerDistanceRate,
                'cornerRefinementMethod': self.detector_params.cornerRefinementMethod,
                'cornerRefinementWinSize': self.detector_params.cornerRefinementWinSize,
                'cornerRefinementMaxIterations': self.detector_params.cornerRefinementMaxIterations,
                'cornerRefinementMinAccuracy': self.detector_params.cornerRefinementMinAccuracy,
                'markerBorderBits': self.detector_params.markerBorderBits,
                'perspectiveRemovePixelPerCell': self.detector_params.perspectiveRemovePixelPerCell,
                'perspectiveRemoveIgnoredMarginPerCell': self.detector_params.perspectiveRemoveIgnoredMarginPerCell,
                'maxErroneousBitsInBorderRate': self.detector_params.maxErroneousBitsInBorderRate,
                'minOtsuStdDev': self.detector_params.minOtsuStdDev,
                'errorCorrectionRate': self.detector_params.errorCorrectionRate
            }
        }