"""
ATC Scoring Module
Calculates Animal Type Classification (ATC) scores based on measurements and characteristics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ATCScoreFactors:
    """Data class for ATC scoring factors."""
    body_conformation: float
    muscle_development: float
    bone_structure: float
    overall_balance: float
    breed_characteristics: float
    health_indicators: float

class ATCScorer:
    """Calculates ATC scores for cattle and buffaloes based on multiple factors."""
    
    def __init__(self):
        """Initialize the ATC scorer with scoring criteria."""
        self.scoring_criteria = {
            'cattle': {
                'height_range': (100, 180),  # cm
                'length_range': (150, 250),  # cm
                'girth_range': (140, 200),   # cm
                'weight_ranges': {
                    'A+': (400, 600),  # kg
                    'A': (350, 500),
                    'B+': (300, 450),
                    'B': (250, 400),
                    'C': (200, 350)
                }
            },
            'buffalo': {
                'height_range': (120, 160),  # cm
                'length_range': (160, 220),  # cm
                'girth_range': (160, 220),   # cm
                'weight_ranges': {
                    'A+': (500, 700),  # kg
                    'A': (450, 600),
                    'B+': (400, 550),
                    'B': (350, 500),
                    'C': (300, 450)
                }
            }
        }
        
        # ATC Grade thresholds
        self.grade_thresholds = {
            'A+': 90,
            'A': 80,
            'B+': 70,
            'B': 60,
            'C': 50,
            'D': 40
        }
    
    def calculate_atc_score(self, measurements: Dict, 
                          keypoints: List[Dict],
                          breed: Optional[str] = None,
                          animal_type: str = 'cattle') -> Dict:
        """
        Calculate comprehensive ATC score.
        
        Args:
            measurements: Dictionary of body measurements
            keypoints: List of detected keypoints
            breed: Detected breed (optional)
            animal_type: Type of animal ('cattle' or 'buffalo')
            
        Returns:
            Dictionary containing ATC score and details
        """
        try:
            # Calculate individual scoring factors
            factors = self._calculate_scoring_factors(
                measurements, keypoints, breed, animal_type
            )
            
            # Calculate weighted overall score
            weights = self._get_scoring_weights(animal_type)
            overall_score = self._calculate_weighted_score(factors, weights)
            
            # Determine ATC grade
            grade = self._determine_atc_grade(overall_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(factors, overall_score, animal_type)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(measurements, keypoints)
            
            return {
                'score': round(overall_score, 2),
                'grade': grade,
                'factors': {
                    'body_conformation': round(factors.body_conformation, 2),
                    'muscle_development': round(factors.muscle_development, 2),
                    'bone_structure': round(factors.bone_structure, 2),
                    'overall_balance': round(factors.overall_balance, 2),
                    'breed_characteristics': round(factors.breed_characteristics, 2),
                    'health_indicators': round(factors.health_indicators, 2)
                },
                'recommendations': recommendations,
                'confidence': confidence,
                'animal_type': animal_type,
                'breed': breed,
                'scoring_weights': weights
            }
            
        except Exception as e:
            logger.error(f"Error calculating ATC score: {e}")
            return {
                'score': 0.0,
                'grade': 'D',
                'factors': {},
                'recommendations': ['Unable to calculate ATC score due to insufficient data'],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_scoring_factors(self, measurements: Dict, 
                                 keypoints: List[Dict], 
                                 breed: Optional[str],
                                 animal_type: str) -> ATCScoreFactors:
        """Calculate individual scoring factors."""
        
        # Body Conformation (30% weight)
        body_conformation = self._score_body_conformation(measurements, animal_type)
        
        # Muscle Development (25% weight)
        muscle_development = self._score_muscle_development(measurements, keypoints)
        
        # Bone Structure (20% weight)
        bone_structure = self._score_bone_structure(measurements, keypoints)
        
        # Overall Balance (15% weight)
        overall_balance = self._score_overall_balance(measurements, keypoints)
        
        # Breed Characteristics (5% weight)
        breed_characteristics = self._score_breed_characteristics(breed, animal_type)
        
        # Health Indicators (5% weight)
        health_indicators = self._score_health_indicators(measurements, keypoints)
        
        return ATCScoreFactors(
            body_conformation=body_conformation,
            muscle_development=muscle_development,
            bone_structure=bone_structure,
            overall_balance=overall_balance,
            breed_characteristics=breed_characteristics,
            health_indicators=health_indicators
        )
    
    def _score_body_conformation(self, measurements: Dict, animal_type: str) -> float:
        """Score body conformation based on measurements."""
        try:
            criteria = self.scoring_criteria.get(animal_type, self.scoring_criteria['cattle'])
            score = 0.0
            total_weight = 0.0
            
            # Height scoring
            if 'height' in measurements and measurements['height']:
                height = measurements['height']
                height_min, height_max = criteria['height_range']
                
                if height_min <= height <= height_max:
                    height_score = 100.0
                else:
                    # Penalize for being outside range
                    if height < height_min:
                        height_score = max(0, 100 - (height_min - height) * 2)
                    else:
                        height_score = max(0, 100 - (height - height_max) * 2)
                
                score += height_score * 0.3
                total_weight += 0.3
            
            # Length scoring
            if 'length' in measurements and measurements['length']:
                length = measurements['length']
                length_min, length_max = criteria['length_range']
                
                if length_min <= length <= length_max:
                    length_score = 100.0
                else:
                    if length < length_min:
                        length_score = max(0, 100 - (length_min - length) * 1.5)
                    else:
                        length_score = max(0, 100 - (length - length_max) * 1.5)
                
                score += length_score * 0.3
                total_weight += 0.3
            
            # Girth scoring
            if 'girth' in measurements and measurements['girth']:
                girth = measurements['girth']
                girth_min, girth_max = criteria['girth_range']
                
                if girth_min <= girth <= girth_max:
                    girth_score = 100.0
                else:
                    if girth < girth_min:
                        girth_score = max(0, 100 - (girth_min - girth) * 1.5)
                    else:
                        girth_score = max(0, 100 - (girth - girth_max) * 1.5)
                
                score += girth_score * 0.4
                total_weight += 0.4
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error scoring body conformation: {e}")
            return 0.0
    
    def _score_muscle_development(self, measurements: Dict, keypoints: List[Dict]) -> float:
        """Score muscle development based on measurements and keypoints."""
        try:
            score = 0.0
            total_weight = 0.0
            
            # Width to height ratio (muscle development indicator)
            if 'width' in measurements and 'height' in measurements:
                if measurements['width'] and measurements['height']:
                    ratio = measurements['width'] / measurements['height']
                    # Optimal ratio is around 0.4-0.6
                    if 0.4 <= ratio <= 0.6:
                        ratio_score = 100.0
                    else:
                        ratio_score = max(0, 100 - abs(ratio - 0.5) * 200)
                    
                    score += ratio_score * 0.4
                    total_weight += 0.4
            
            # Girth to length ratio (body depth)
            if 'girth' in measurements and 'length' in measurements:
                if measurements['girth'] and measurements['length']:
                    ratio = measurements['girth'] / measurements['length']
                    # Optimal ratio is around 0.7-0.9
                    if 0.7 <= ratio <= 0.9:
                        girth_score = 100.0
                    else:
                        girth_score = max(0, 100 - abs(ratio - 0.8) * 100)
                    
                    score += girth_score * 0.6
                    total_weight += 0.6
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error scoring muscle development: {e}")
            return 0.0
    
    def _score_bone_structure(self, measurements: Dict, keypoints: List[Dict]) -> float:
        """Score bone structure based on keypoint analysis."""
        try:
            if not keypoints:
                return 0.0
            
            # Analyze keypoint symmetry and alignment
            symmetry_score = self._analyze_symmetry(keypoints)
            
            # Analyze bone proportions
            proportion_score = self._analyze_bone_proportions(measurements)
            
            # Combine scores
            return (symmetry_score * 0.6 + proportion_score * 0.4)
            
        except Exception as e:
            logger.warning(f"Error scoring bone structure: {e}")
            return 0.0
    
    def _score_overall_balance(self, measurements: Dict, keypoints: List[Dict]) -> float:
        """Score overall body balance and proportions."""
        try:
            score = 0.0
            total_weight = 0.0
            
            # Height to length ratio
            if 'height' in measurements and 'length' in measurements:
                if measurements['height'] and measurements['length']:
                    ratio = measurements['height'] / measurements['length']
                    # Optimal ratio is around 0.6-0.8
                    if 0.6 <= ratio <= 0.8:
                        balance_score = 100.0
                    else:
                        balance_score = max(0, 100 - abs(ratio - 0.7) * 100)
                    
                    score += balance_score * 0.5
                    total_weight += 0.5
            
            # Keypoint alignment analysis
            if keypoints:
                alignment_score = self._analyze_keypoint_alignment(keypoints)
                score += alignment_score * 0.5
                total_weight += 0.5
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error scoring overall balance: {e}")
            return 0.0
    
    def _score_breed_characteristics(self, breed: Optional[str], animal_type: str) -> float:
        """Score breed-specific characteristics."""
        if not breed:
            return 50.0  # Neutral score if breed unknown
        
        # Breed-specific scoring (simplified)
        breed_scores = {
            'cattle': {
                'gir': 85, 'sahival': 80, 'red_sindhi': 75, 'haryana': 70,
                'kankrej': 80, 'tharparkar': 75, 'ongole': 70, 'hallikar': 75,
                'amritmahal': 70, 'kangayam': 75
            },
            'buffalo': {
                'murrah': 90, 'nili_ravi': 85, 'jafrabadi': 80, 'surti': 75,
                'mehsana': 80, 'banni': 75
            }
        }
        
        return breed_scores.get(animal_type, {}).get(breed.lower(), 50.0)
    
    def _score_health_indicators(self, measurements: Dict, keypoints: List[Dict]) -> float:
        """Score health indicators based on measurements and keypoints."""
        try:
            score = 0.0
            total_weight = 0.0
            
            # Body condition assessment
            if 'girth' in measurements and 'length' in measurements:
                if measurements['girth'] and measurements['length']:
                    bcs_ratio = measurements['girth'] / measurements['length']
                    # Healthy ratio is around 0.7-0.9
                    if 0.7 <= bcs_ratio <= 0.9:
                        health_score = 100.0
                    else:
                        health_score = max(0, 100 - abs(bcs_ratio - 0.8) * 100)
                    
                    score += health_score * 0.6
                    total_weight += 0.6
            
            # Keypoint quality (indicates good health/condition)
            if keypoints:
                avg_confidence = np.mean([kp['confidence'] for kp in keypoints])
                quality_score = avg_confidence * 100
                
                score += quality_score * 0.4
                total_weight += 0.4
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error scoring health indicators: {e}")
            return 0.0
    
    def _analyze_symmetry(self, keypoints: List[Dict]) -> float:
        """Analyze bilateral symmetry of keypoints."""
        try:
            # Define symmetric point pairs
            symmetric_pairs = [
                ('left_eye', 'right_eye'),
                ('left_ear', 'right_ear'),
                ('shoulder_left', 'shoulder_right'),
                ('hip_left', 'hip_right'),
                ('knee_left', 'knee_right')
            ]
            
            symmetry_scores = []
            for left_name, right_name in symmetric_pairs:
                left_kp = next((kp for kp in keypoints if kp['name'] == left_name), None)
                right_kp = next((kp for kp in keypoints if kp['name'] == right_name), None)
                
                if left_kp and right_kp:
                    # Calculate vertical symmetry
                    y_diff = abs(left_kp['y'] - right_kp['y'])
                    symmetry_score = max(0, 100 - y_diff * 1000)  # Scale factor
                    symmetry_scores.append(symmetry_score)
            
            return np.mean(symmetry_scores) if symmetry_scores else 50.0
            
        except Exception as e:
            logger.warning(f"Error analyzing symmetry: {e}")
            return 50.0
    
    def _analyze_bone_proportions(self, measurements: Dict) -> float:
        """Analyze bone structure proportions."""
        try:
            if not measurements:
                return 50.0
            
            # Check for reasonable proportions
            score = 100.0
            
            # Height should be reasonable relative to length
            if 'height' in measurements and 'length' in measurements:
                if measurements['height'] and measurements['length']:
                    ratio = measurements['height'] / measurements['length']
                    if ratio < 0.4 or ratio > 1.0:  # Unreasonable proportions
                        score -= 20
            
            # Width should be reasonable relative to height
            if 'width' in measurements and 'height' in measurements:
                if measurements['width'] and measurements['height']:
                    ratio = measurements['width'] / measurements['height']
                    if ratio < 0.2 or ratio > 0.8:  # Unreasonable proportions
                        score -= 20
            
            return max(0, score)
            
        except Exception as e:
            logger.warning(f"Error analyzing bone proportions: {e}")
            return 50.0
    
    def _analyze_keypoint_alignment(self, keypoints: List[Dict]) -> float:
        """Analyze keypoint alignment and posture."""
        try:
            # Check spine alignment
            spine_points = [kp for kp in keypoints if 'spine' in kp['name']]
            if len(spine_points) >= 3:
                # Calculate spine straightness
                x_coords = [kp['x'] for kp in spine_points]
                y_coords = [kp['y'] for kp in spine_points]
                
                # Simple linear regression to check alignment
                if len(x_coords) > 1:
                    slope = np.polyfit(x_coords, y_coords, 1)[0]
                    alignment_score = max(0, 100 - abs(slope) * 50)
                    return alignment_score
            
            return 50.0
            
        except Exception as e:
            logger.warning(f"Error analyzing keypoint alignment: {e}")
            return 50.0
    
    def _calculate_weighted_score(self, factors: ATCScoreFactors, weights: Dict) -> float:
        """Calculate weighted overall score."""
        score = (
            factors.body_conformation * weights['body_conformation'] +
            factors.muscle_development * weights['muscle_development'] +
            factors.bone_structure * weights['bone_structure'] +
            factors.overall_balance * weights['overall_balance'] +
            factors.breed_characteristics * weights['breed_characteristics'] +
            factors.health_indicators * weights['health_indicators']
        )
        return min(100.0, max(0.0, score))
    
    def _get_scoring_weights(self, animal_type: str) -> Dict:
        """Get scoring weights for different animal types."""
        return {
            'body_conformation': 0.30,
            'muscle_development': 0.25,
            'bone_structure': 0.20,
            'overall_balance': 0.15,
            'breed_characteristics': 0.05,
            'health_indicators': 0.05
        }
    
    def _determine_atc_grade(self, score: float) -> str:
        """Determine ATC grade based on score."""
        for grade, threshold in sorted(self.grade_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return grade
        return 'D'
    
    def _generate_recommendations(self, factors: ATCScoreFactors, 
                                overall_score: float, animal_type: str) -> List[str]:
        """Generate improvement recommendations based on scoring factors."""
        recommendations = []
        
        if factors.body_conformation < 70:
            recommendations.append("Improve body conformation through better nutrition and breeding")
        
        if factors.muscle_development < 70:
            recommendations.append("Enhance muscle development with appropriate exercise and diet")
        
        if factors.bone_structure < 70:
            recommendations.append("Address bone structure issues - consider veterinary consultation")
        
        if factors.overall_balance < 70:
            recommendations.append("Work on overall body balance and proportions")
        
        if factors.health_indicators < 70:
            recommendations.append("Improve health indicators - check for underlying health issues")
        
        if overall_score >= 90:
            recommendations.append("Excellent ATC score - maintain current care and breeding practices")
        elif overall_score >= 80:
            recommendations.append("Good ATC score - minor improvements possible")
        elif overall_score >= 70:
            recommendations.append("Average ATC score - significant improvement potential")
        else:
            recommendations.append("Below average ATC score - comprehensive improvement needed")
        
        return recommendations
    
    def _calculate_confidence(self, measurements: Dict, keypoints: List[Dict]) -> float:
        """Calculate confidence in the ATC score based on data quality."""
        try:
            confidence_factors = []
            
            # Measurement completeness
            required_measurements = ['height', 'length', 'girth', 'width']
            measurement_completeness = sum(1 for m in required_measurements 
                                        if m in measurements and measurements[m] is not None)
            measurement_confidence = measurement_completeness / len(required_measurements)
            confidence_factors.append(measurement_confidence)
            
            # Keypoint quality
            if keypoints:
                avg_keypoint_confidence = np.mean([kp['confidence'] for kp in keypoints])
                confidence_factors.append(avg_keypoint_confidence)
            else:
                confidence_factors.append(0.0)
            
            # Data consistency
            consistency_score = self._check_data_consistency(measurements)
            confidence_factors.append(consistency_score)
            
            return float(np.mean(confidence_factors))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.0
    
    def _check_data_consistency(self, measurements: Dict) -> float:
        """Check consistency of measurement data."""
        try:
            if not measurements:
                return 0.0
            
            # Check for reasonable relationships between measurements
            consistency_score = 100.0
            
            # Height should be less than length
            if 'height' in measurements and 'length' in measurements:
                if measurements['height'] and measurements['length']:
                    if measurements['height'] > measurements['length']:
                        consistency_score -= 30
            
            # Girth should be reasonable relative to length
            if 'girth' in measurements and 'length' in measurements:
                if measurements['girth'] and measurements['length']:
                    ratio = measurements['girth'] / measurements['length']
                    if ratio > 1.2:  # Girth shouldn't be much larger than length
                        consistency_score -= 20
            
            return max(0, consistency_score) / 100.0
            
        except Exception as e:
            logger.warning(f"Error checking data consistency: {e}")
            return 0.0
