"""
Measurement Calculator Module
Calculates body measurements from keypoints and ArUco markers
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MeasurementCalculator:
    """Calculates body measurements from keypoints and scale references."""
    
    def __init__(self, scale_factor: float = 1.0):
        """
        Initialize the measurement calculator.
        
        Args:
            scale_factor: Scale factor in cm per pixel
        """
        self.scale_factor = scale_factor
        self.units = 'cm'
        
        # Measurement definitions
        self.measurement_definitions = {
            'height': {
                'description': 'Height from ground to top of head',
                'keypoints': ['spine_upper', 'spine_lower'],
                'method': 'vertical_distance'
            },
            'length': {
                'description': 'Body length from shoulder to hip',
                'keypoints': ['left_shoulder', 'left_hip'],
                'method': 'horizontal_distance'
            },
            'width': {
                'description': 'Body width at shoulders',
                'keypoints': ['left_shoulder', 'right_shoulder'],
                'method': 'horizontal_distance'
            },
            'girth': {
                'description': 'Chest girth (circumference)',
                'keypoints': ['chest_center', 'left_shoulder', 'right_shoulder'],
                'method': 'circumference_estimate'
            },
            'leg_length': {
                'description': 'Leg length from hip to ankle',
                'keypoints': ['left_hip', 'left_ankle'],
                'method': 'vertical_distance'
            },
            'head_length': {
                'description': 'Head length from nose to ear',
                'keypoints': ['nose', 'left_ear'],
                'method': 'euclidean_distance'
            },
            'tail_length': {
                'description': 'Tail length from base to tip',
                'keypoints': ['tail_base', 'tail_tip'],
                'method': 'euclidean_distance'
            }
        }
        
        logger.info(f"Measurement calculator initialized with scale factor: {scale_factor} cm/px")
    
    def set_scale_factor(self, scale_factor: float):
        """Set the scale factor for measurements."""
        self.scale_factor = scale_factor
        logger.info(f"Scale factor updated to: {scale_factor} cm/px")
    
    def calculate_measurements(self, keypoints: List[Dict], 
                             aruco_markers: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate body measurements from keypoints.
        
        Args:
            keypoints: List of detected keypoints
            aruco_markers: Optional ArUco markers for scale reference
            
        Returns:
            Dictionary containing measurement results
        """
        try:
            # Update scale factor from ArUco markers if available
            if aruco_markers:
                scale_factor = self._calculate_scale_from_markers(aruco_markers)
                if scale_factor > 0:
                    self.set_scale_factor(scale_factor)
            
            measurements = {}
            measurement_confidence = []
            
            # Calculate each defined measurement
            for measurement_name, definition in self.measurement_definitions.items():
                try:
                    value, confidence = self._calculate_single_measurement(
                        keypoints, definition
                    )
                    
                    if value is not None:
                        measurements[measurement_name] = value
                        measurement_confidence.append(confidence)
                    
                except Exception as e:
                    logger.warning(f"Error calculating {measurement_name}: {e}")
                    continue
            
            # Calculate overall confidence
            overall_confidence = np.mean(measurement_confidence) if measurement_confidence else 0.0
            
            # Add additional calculated measurements
            additional_measurements = self._calculate_additional_measurements(keypoints)
            measurements.update(additional_measurements)
            
            return {
                'measurements': measurements,
                'confidence': overall_confidence,
                'scale_factor': self.scale_factor,
                'units': self.units,
                'valid_measurements': len(measurements),
                'total_measurements': len(self.measurement_definitions),
                'measurement_details': self._get_measurement_details(measurements)
            }
            
        except Exception as e:
            logger.error(f"Error calculating measurements: {e}")
            return {
                'measurements': {},
                'confidence': 0.0,
                'scale_factor': self.scale_factor,
                'units': self.units,
                'valid_measurements': 0,
                'error': str(e)
            }
    
    def _calculate_single_measurement(self, keypoints: List[Dict], 
                                    definition: Dict) -> Tuple[Optional[float], float]:
        """
        Calculate a single measurement.
        
        Args:
            keypoints: List of keypoints
            definition: Measurement definition
            
        Returns:
            Tuple of (value, confidence)
        """
        try:
            required_keypoints = definition['keypoints']
            method = definition['method']
            
            # Find required keypoints
            kp_dict = {kp['name']: kp for kp in keypoints}
            required_kps = [kp_dict.get(name) for name in required_keypoints]
            
            # Check if all required keypoints are available
            missing_kps = [name for name, kp in zip(required_keypoints, required_kps) if kp is None]
            if missing_kps:
                logger.debug(f"Missing keypoints for measurement: {missing_kps}")
                return None, 0.0
            
            # Calculate measurement based on method
            if method == 'euclidean_distance':
                value = self._calculate_euclidean_distance(required_kps[0], required_kps[1])
            elif method == 'horizontal_distance':
                value = self._calculate_horizontal_distance(required_kps[0], required_kps[1])
            elif method == 'vertical_distance':
                value = self._calculate_vertical_distance(required_kps[0], required_kps[1])
            elif method == 'circumference_estimate':
                value = self._calculate_circumference_estimate(required_kps)
            else:
                logger.warning(f"Unknown measurement method: {method}")
                return None, 0.0
            
            # Calculate confidence based on keypoint confidences
            confidences = [kp['confidence'] for kp in required_kps]
            confidence = np.mean(confidences) if confidences else 0.0
            
            return value, confidence
            
        except Exception as e:
            logger.warning(f"Error calculating single measurement: {e}")
            return None, 0.0
    
    def _calculate_euclidean_distance(self, kp1: Dict, kp2: Dict) -> float:
        """Calculate Euclidean distance between two keypoints."""
        try:
            dx = kp1['x'] - kp2['x']
            dy = kp1['y'] - kp2['y']
            distance_pixels = np.sqrt(dx**2 + dy**2)
            return distance_pixels * self.scale_factor
        except Exception as e:
            logger.warning(f"Error calculating euclidean distance: {e}")
            return 0.0
    
    def _calculate_horizontal_distance(self, kp1: Dict, kp2: Dict) -> float:
        """Calculate horizontal distance between two keypoints."""
        try:
            dx = abs(kp1['x'] - kp2['x'])
            return dx * self.scale_factor
        except Exception as e:
            logger.warning(f"Error calculating horizontal distance: {e}")
            return 0.0
    
    def _calculate_vertical_distance(self, kp1: Dict, kp2: Dict) -> float:
        """Calculate vertical distance between two keypoints."""
        try:
            dy = abs(kp1['y'] - kp2['y'])
            return dy * self.scale_factor
        except Exception as e:
            logger.warning(f"Error calculating vertical distance: {e}")
            return 0.0
    
    def _calculate_circumference_estimate(self, keypoints: List[Dict]) -> float:
        """Estimate circumference from keypoints."""
        try:
            if len(keypoints) < 3:
                return 0.0
            
            # Use chest center and shoulder points to estimate girth
            center = keypoints[0]  # chest_center
            left_shoulder = keypoints[1]  # left_shoulder
            right_shoulder = keypoints[2]  # right_shoulder
            
            # Calculate radius from center to shoulders
            left_radius = self._calculate_euclidean_distance(center, left_shoulder)
            right_radius = self._calculate_euclidean_distance(center, right_shoulder)
            avg_radius = (left_radius + right_radius) / 2
            
            # Estimate circumference as 2 * pi * radius
            circumference = 2 * np.pi * avg_radius
            return circumference
            
        except Exception as e:
            logger.warning(f"Error calculating circumference estimate: {e}")
            return 0.0
    
    def _calculate_additional_measurements(self, keypoints: List[Dict]) -> Dict:
        """Calculate additional derived measurements."""
        try:
            additional = {}
            
            # Body mass index estimate
            if 'height' in self.measurements and 'girth' in self.measurements:
                height = self.measurements['height']
                girth = self.measurements['girth']
                if height > 0 and girth > 0:
                    # Simple BMI estimate for cattle
                    additional['bmi_estimate'] = (girth**2 * height) / 10000
            
            # Body condition score estimate
            bcs = self._estimate_body_condition_score(keypoints)
            if bcs is not None:
                additional['body_condition_score'] = bcs
            
            # Symmetry score
            symmetry = self._calculate_symmetry_score(keypoints)
            if symmetry is not None:
                additional['symmetry_score'] = symmetry
            
            return additional
            
        except Exception as e:
            logger.warning(f"Error calculating additional measurements: {e}")
            return {}
    
    def _estimate_body_condition_score(self, keypoints: List[Dict]) -> Optional[float]:
        """Estimate body condition score from keypoints."""
        try:
            # This is a simplified BCS estimation
            # In practice, this would require more sophisticated analysis
            
            # Look for key indicators of body condition
            indicators = []
            
            # Check for visible ribs (indicates low BCS)
            rib_keypoints = [kp for kp in keypoints if 'rib' in kp['name']]
            if rib_keypoints:
                # More visible ribs = lower BCS
                rib_visibility = np.mean([kp['confidence'] for kp in rib_keypoints])
                indicators.append(5 - rib_visibility * 4)  # Scale to 1-5
            
            # Check for hip prominence
            hip_keypoints = [kp for kp in keypoints if 'hip' in kp['name']]
            if hip_keypoints:
                hip_confidence = np.mean([kp['confidence'] for kp in hip_keypoints])
                indicators.append(hip_confidence * 5)  # Scale to 1-5
            
            if indicators:
                return np.mean(indicators)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error estimating body condition score: {e}")
            return None
    
    def _calculate_symmetry_score(self, keypoints: List[Dict]) -> Optional[float]:
        """Calculate body symmetry score."""
        try:
            # Define symmetric point pairs
            symmetric_pairs = [
                ('left_eye', 'right_eye'),
                ('left_ear', 'right_ear'),
                ('left_shoulder', 'right_shoulder'),
                ('left_hip', 'right_hip'),
                ('left_knee', 'right_knee')
            ]
            
            symmetry_scores = []
            
            for left_name, right_name in symmetric_pairs:
                left_kp = next((kp for kp in keypoints if kp['name'] == left_name), None)
                right_kp = next((kp for kp in keypoints if kp['name'] == right_name), None)
                
                if left_kp and right_kp:
                    # Calculate vertical symmetry
                    y_diff = abs(left_kp['y'] - right_kp['y'])
                    # Normalize by image height (assuming keypoints are in image coordinates)
                    symmetry_score = max(0, 1 - y_diff / 100)  # Adjust normalization as needed
                    symmetry_scores.append(symmetry_score)
            
            if symmetry_scores:
                return np.mean(symmetry_scores)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating symmetry score: {e}")
            return None
    
    def _calculate_scale_from_markers(self, aruco_markers: List[Dict]) -> float:
        """Calculate scale factor from ArUco markers."""
        try:
            if not aruco_markers:
                return 0.0
            
            # Use the marker with the highest confidence (largest size)
            best_marker = max(aruco_markers, key=lambda m: m.get('pixel_size', 0))
            
            if best_marker.get('pixel_size', 0) > 0:
                # Scale factor is already calculated in the marker
                return best_marker.get('scale_factor', 0.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating scale from markers: {e}")
            return 0.0
    
    def _get_measurement_details(self, measurements: Dict) -> Dict:
        """Get detailed information about measurements."""
        try:
            details = {}
            
            for measurement_name, value in measurements.items():
                if measurement_name in self.measurement_definitions:
                    definition = self.measurement_definitions[measurement_name]
                    details[measurement_name] = {
                        'value': value,
                        'description': definition['description'],
                        'units': self.units,
                        'method': definition['method']
                    }
                else:
                    # Additional measurement
                    details[measurement_name] = {
                        'value': value,
                        'description': f'Calculated {measurement_name}',
                        'units': self.units,
                        'method': 'derived'
                    }
            
            return details
            
        except Exception as e:
            logger.warning(f"Error getting measurement details: {e}")
            return {}
    
    def validate_measurements(self, measurements: Dict) -> Dict:
        """Validate measurements for reasonableness."""
        try:
            validation = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'quality_score': 0.0
            }
            
            # Define reasonable ranges for cattle measurements (in cm)
            reasonable_ranges = {
                'height': (80, 200),
                'length': (120, 250),
                'width': (40, 80),
                'girth': (120, 220),
                'leg_length': (60, 120),
                'head_length': (30, 60),
                'tail_length': (20, 80)
            }
            
            quality_factors = []
            
            for measurement_name, value in measurements.items():
                if measurement_name in reasonable_ranges:
                    min_val, max_val = reasonable_ranges[measurement_name]
                    
                    if value < min_val:
                        validation['warnings'].append(f'{measurement_name} seems too small ({value:.1f} cm)')
                        quality_factors.append(0.5)
                    elif value > max_val:
                        validation['warnings'].append(f'{measurement_name} seems too large ({value:.1f} cm)')
                        quality_factors.append(0.5)
                    else:
                        quality_factors.append(1.0)
            
            # Calculate quality score
            if quality_factors:
                validation['quality_score'] = np.mean(quality_factors)
            
            # Check for missing critical measurements
            critical_measurements = ['height', 'length', 'girth']
            missing_critical = [m for m in critical_measurements if m not in measurements]
            
            if missing_critical:
                validation['warnings'].append(f'Missing critical measurements: {missing_critical}')
                validation['quality_score'] *= 0.7
            
            return validation
            
        except Exception as e:
            logger.warning(f"Error validating measurements: {e}")
            return {
                'valid': False,
                'warnings': [],
                'errors': [f'Validation error: {str(e)}'],
                'quality_score': 0.0
            }
    
    def export_measurements(self, measurements: Dict, format: str = 'json') -> str:
        """Export measurements in specified format."""
        try:
            if format.lower() == 'json':
                import json
                return json.dumps(measurements, indent=2)
            elif format.lower() == 'csv':
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(['Measurement', 'Value', 'Units'])
                
                # Write data
                for name, value in measurements.items():
                    writer.writerow([name, value, self.units])
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting measurements: {e}")
            return str(e)
    
    def get_calculator_status(self) -> Dict:
        """Get calculator status and configuration."""
        return {
            'status': 'operational',
            'scale_factor': self.scale_factor,
            'units': self.units,
            'measurement_definitions': len(self.measurement_definitions),
            'capabilities': {
                'euclidean_distance': True,
                'horizontal_distance': True,
                'vertical_distance': True,
                'circumference_estimate': True,
                'body_condition_score': True,
                'symmetry_analysis': True,
                'validation': True,
                'export': True
            }
        }