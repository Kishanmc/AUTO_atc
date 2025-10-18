"""
ATC Pipeline Service
Orchestrates the complete ATC analysis pipeline
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path

# Import AI modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_modules.detection.animal_detector import AnimalDetector
from ai_modules.detection.keypoint_detector import KeypointDetector
from ai_modules.detection.aruco_detector import ArucoDetector
from ai_modules.measurement.calculator import MeasurementCalculator
from ai_modules.scoring.atc_scorer import ATCScorer

logger = logging.getLogger(__name__)

class ATCPipeline:
    """Main pipeline for Animal Type Classification analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ATC pipeline.
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config or self._get_default_config()
        
        # Initialize AI modules
        self.animal_detector = AnimalDetector(
            model_path=self.config.get('detection_model_path', 'yolov8n.pt'),
            confidence_threshold=self.config.get('detection_confidence', 0.5)
        )
        
        self.keypoint_detector = KeypointDetector(
            model_complexity=self.config.get('keypoint_complexity', 1),
            min_detection_confidence=self.config.get('keypoint_confidence', 0.5)
        )
        
        self.aruco_detector = ArucoDetector(
            dictionary_type=self.config.get('aruco_dictionary', 6),
            marker_size=self.config.get('aruco_marker_size', 5.0)
        )
        
        self.measurement_calculator = MeasurementCalculator(
            scale_factor=self.config.get('scale_factor', 1.0)
        )
        
        self.atc_scorer = ATCScorer()
        
        logger.info("ATC Pipeline initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration."""
        return {
            'detection_model_path': 'yolov8n.pt',
            'detection_confidence': 0.5,
            'keypoint_complexity': 1,
            'keypoint_confidence': 0.5,
            'aruco_dictionary': 6,
            'aruco_marker_size': 5.0,
            'scale_factor': 1.0,
            'enable_aruco_detection': True,
            'enable_measurements': True,
            'enable_atc_scoring': True
        }
    
    def detect_animal(self, image: np.ndarray) -> Dict:
        """
        Detect animal in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self.animal_detector.preprocess_image(image)
            
            # Detect animals
            detection_result = self.animal_detector.detect_animals(processed_image)
            
            processing_time = time.time() - start_time
            detection_result['processing_time'] = processing_time
            
            logger.info(f"Animal detection completed in {processing_time:.2f}s")
            return detection_result
            
        except Exception as e:
            logger.error(f"Error in animal detection: {e}")
            return {
                'detected': False,
                'detections': [],
                'best_detection': None,
                'confidence': 0.0,
                'bounding_box': None,
                'animal_type': 'unknown',
                'processing_time': 0.0,
                'error': str(e)
            }
    
    def detect_keypoints(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> Dict:
        """
        Detect keypoints on the animal.
        
        Args:
            image: Input image as numpy array
            bbox: Optional bounding box to focus detection
            
        Returns:
            Dictionary containing keypoint detection results
        """
        try:
            start_time = time.time()
            
            # Detect keypoints
            keypoint_result = self.keypoint_detector.detect_keypoints(image, bbox)
            
            processing_time = time.time() - start_time
            keypoint_result['processing_time'] = processing_time
            
            logger.info(f"Keypoint detection completed in {processing_time:.2f}s")
            return keypoint_result
            
        except Exception as e:
            logger.error(f"Error in keypoint detection: {e}")
            return {
                'detected': False,
                'keypoints': [],
                'confidence': 0.0,
                'landmarks': None,
                'image_shape': image.shape,
                'processing_time': 0.0,
                'error': str(e)
            }
    
    def detect_aruco_markers(self, image: np.ndarray) -> Dict:
        """
        Detect ArUco markers for scale reference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing ArUco detection results
        """
        try:
            if not self.config.get('enable_aruco_detection', True):
                return {
                    'detected': False,
                    'markers': [],
                    'count': 0,
                    'scale_factor': 1.0
                }
            
            start_time = time.time()
            
            # Detect ArUco markers
            aruco_result = self.aruco_detector.detect_markers(image)
            
            # Calculate scale factor if markers found
            scale_factor = 1.0
            if aruco_result['detected']:
                scale_factor = self.aruco_detector.calculate_scale_factor(aruco_result['markers'])
                self.measurement_calculator.set_scale_factor(scale_factor)
            
            processing_time = time.time() - start_time
            aruco_result['processing_time'] = processing_time
            aruco_result['scale_factor'] = scale_factor
            
            logger.info(f"ArUco detection completed in {processing_time:.2f}s")
            return aruco_result
            
        except Exception as e:
            logger.error(f"Error in ArUco detection: {e}")
            return {
                'detected': False,
                'markers': [],
                'count': 0,
                'scale_factor': 1.0,
                'processing_time': 0.0,
                'error': str(e)
            }
    
    def calculate_measurements(self, image: np.ndarray, 
                             keypoints: List[Dict],
                             aruco_markers: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate body measurements from keypoints.
        
        Args:
            image: Input image as numpy array
            keypoints: List of detected keypoints
            aruco_markers: Optional ArUco markers for scale reference
            
        Returns:
            Dictionary containing measurement results
        """
        try:
            if not self.config.get('enable_measurements', True):
                return {
                    'measurements': {},
                    'confidence': 0.0,
                    'scale_factor': 1.0,
                    'units': 'cm',
                    'valid_measurements': 0
                }
            
            start_time = time.time()
            
            # Calculate measurements
            measurement_result = self.measurement_calculator.calculate_measurements(
                keypoints, aruco_markers
            )
            
            processing_time = time.time() - start_time
            measurement_result['processing_time'] = processing_time
            
            logger.info(f"Measurement calculation completed in {processing_time:.2f}s")
            return measurement_result
            
        except Exception as e:
            logger.error(f"Error in measurement calculation: {e}")
            return {
                'measurements': {},
                'confidence': 0.0,
                'scale_factor': 1.0,
                'units': 'cm',
                'valid_measurements': 0,
                'processing_time': 0.0,
                'error': str(e)
            }
    
    def calculate_atc_score(self, image: np.ndarray,
                          keypoints: List[Dict],
                          measurements: Dict,
                          breed: Optional[str] = None,
                          animal_type: str = 'cattle') -> Dict:
        """
        Calculate ATC score based on measurements and characteristics.
        
        Args:
            image: Input image as numpy array
            keypoints: List of detected keypoints
            measurements: Dictionary of body measurements
            breed: Detected breed (optional)
            animal_type: Type of animal ('cattle' or 'buffalo')
            
        Returns:
            Dictionary containing ATC scoring results
        """
        try:
            if not self.config.get('enable_atc_scoring', True):
                return {
                    'score': 0.0,
                    'grade': 'N/A',
                    'factors': {},
                    'recommendations': ['ATC scoring disabled'],
                    'confidence': 0.0
                }
            
            start_time = time.time()
            
            # Calculate ATC score
            atc_result = self.atc_scorer.calculate_atc_score(
                measurements, keypoints, breed, animal_type
            )
            
            processing_time = time.time() - start_time
            atc_result['processing_time'] = processing_time
            
            logger.info(f"ATC scoring completed in {processing_time:.2f}s")
            return atc_result
            
        except Exception as e:
            logger.error(f"Error in ATC scoring: {e}")
            return {
                'score': 0.0,
                'grade': 'D',
                'factors': {},
                'recommendations': ['Error in ATC scoring'],
                'confidence': 0.0,
                'processing_time': 0.0,
                'error': str(e)
            }
    
    def run_complete_analysis(self, image: np.ndarray,
                            breed: Optional[str] = None,
                            animal_type: str = 'cattle') -> Dict:
        """
        Run complete ATC analysis pipeline.
        
        Args:
            image: Input image as numpy array
            breed: Detected breed (optional)
            animal_type: Type of animal ('cattle' or 'buffalo')
            
        Returns:
            Dictionary containing complete analysis results
        """
        try:
            start_time = time.time()
            
            # Step 1: Animal Detection
            logger.info("Starting animal detection...")
            detection_result = self.detect_animal(image)
            
            if not detection_result['detected']:
                return {
                    'success': False,
                    'error': 'No animal detected in image',
                    'detection_result': detection_result,
                    'processing_time': time.time() - start_time
                }
            
            # Step 2: ArUco Marker Detection
            logger.info("Detecting ArUco markers...")
            aruco_result = self.detect_aruco_markers(image)
            
            # Step 3: Keypoint Detection
            logger.info("Detecting keypoints...")
            keypoint_result = self.detect_keypoints(
                image, detection_result['bounding_box']
            )
            
            # Step 4: Measurement Calculation
            logger.info("Calculating measurements...")
            measurement_result = self.calculate_measurements(
                image, keypoint_result['keypoints'], aruco_result['markers']
            )
            
            # Step 5: ATC Scoring
            logger.info("Calculating ATC score...")
            atc_result = self.calculate_atc_score(
                image, keypoint_result['keypoints'], 
                measurement_result['measurements'], breed, animal_type
            )
            
            # Compile results
            total_processing_time = time.time() - start_time
            
            results = {
                'success': True,
                'processing_time': total_processing_time,
                'detection_result': detection_result,
                'aruco_result': aruco_result,
                'keypoint_result': keypoint_result,
                'measurement_result': measurement_result,
                'atc_result': atc_result,
                'summary': {
                    'animal_detected': detection_result['detected'],
                    'keypoints_detected': keypoint_result['detected'],
                    'aruco_markers_detected': aruco_result['detected'],
                    'measurements_calculated': measurement_result['valid_measurements'],
                    'atc_score': atc_result['score'],
                    'atc_grade': atc_result['grade'],
                    'overall_confidence': self._calculate_overall_confidence([
                        detection_result['confidence'],
                        keypoint_result['confidence'],
                        measurement_result['confidence'],
                        atc_result['confidence']
                    ])
                }
            }
            
            logger.info(f"Complete ATC analysis finished in {total_processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
    def _calculate_overall_confidence(self, confidences: List[float]) -> float:
        """Calculate overall confidence from individual module confidences."""
        try:
            # Filter out None values and calculate weighted average
            valid_confidences = [c for c in confidences if c is not None and c > 0]
            
            if not valid_confidences:
                return 0.0
            
            # Weight detection and keypoint detection more heavily
            weights = [0.3, 0.3, 0.2, 0.2]  # detection, keypoints, measurements, atc
            
            weighted_sum = sum(w * c for w, c in zip(weights, valid_confidences))
            total_weight = sum(weights[:len(valid_confidences)])
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating overall confidence: {e}")
            return 0.0
    
    def validate_analysis_data(self, results: Dict) -> Dict:
        """
        Validate the analysis results for consistency and quality.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation_result = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'quality_score': 0.0
            }
            
            # Check if analysis was successful
            if not results.get('success', False):
                validation_result['valid'] = False
                validation_result['errors'].append('Analysis failed')
                return validation_result
            
            # Check detection quality
            detection_result = results.get('detection_result', {})
            if not detection_result.get('detected', False):
                validation_result['valid'] = False
                validation_result['errors'].append('No animal detected')
            elif detection_result.get('confidence', 0) < 0.5:
                validation_result['warnings'].append('Low detection confidence')
            
            # Check keypoint quality
            keypoint_result = results.get('keypoint_result', {})
            if not keypoint_result.get('detected', False):
                validation_result['warnings'].append('No keypoints detected')
            elif keypoint_result.get('confidence', 0) < 0.5:
                validation_result['warnings'].append('Low keypoint detection confidence')
            
            # Check measurement quality
            measurement_result = results.get('measurement_result', {})
            if measurement_result.get('valid_measurements', 0) == 0:
                validation_result['warnings'].append('No valid measurements calculated')
            
            # Check ATC score validity
            atc_result = results.get('atc_result', {})
            if atc_result.get('score', 0) == 0:
                validation_result['warnings'].append('ATC score not calculated')
            
            # Calculate quality score
            quality_factors = []
            
            if detection_result.get('detected', False):
                quality_factors.append(detection_result.get('confidence', 0))
            
            if keypoint_result.get('detected', False):
                quality_factors.append(keypoint_result.get('confidence', 0))
            
            if measurement_result.get('valid_measurements', 0) > 0:
                quality_factors.append(measurement_result.get('confidence', 0))
            
            if atc_result.get('score', 0) > 0:
                quality_factors.append(atc_result.get('confidence', 0))
            
            if quality_factors:
                validation_result['quality_score'] = sum(quality_factors) / len(quality_factors)
            else:
                validation_result['quality_score'] = 0.0
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating analysis data: {e}")
            return {
                'valid': False,
                'warnings': [],
                'errors': [f'Validation error: {str(e)}'],
                'quality_score': 0.0
            }
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status and configuration."""
        return {
            'status': 'operational',
            'config': self.config,
            'modules': {
                'animal_detector': 'loaded',
                'keypoint_detector': 'loaded',
                'aruco_detector': 'loaded',
                'measurement_calculator': 'loaded',
                'atc_scorer': 'loaded'
            },
            'capabilities': {
                'animal_detection': True,
                'keypoint_detection': True,
                'aruco_detection': self.config.get('enable_aruco_detection', True),
                'measurement_calculation': self.config.get('enable_measurements', True),
                'atc_scoring': self.config.get('enable_atc_scoring', True)
            }
        }
