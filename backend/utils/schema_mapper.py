"""
Schema mapping utilities for BPA integration
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BPASchemaMapper:
    """Maps AutoATC data to BPA schema format."""
    
    def __init__(self):
        """Initialize the BPA schema mapper."""
        self.bpa_field_mapping = {
            'animal_id': 'animalId',
            'analysis_date': 'analysisDate',
            'atc_score': 'atcScore',
            'atc_grade': 'atcGrade',
            'breed': 'breed',
            'measurements': 'measurements',
            'diseases': 'healthStatus',
            'confidence': 'confidenceScore',
            'image_path': 'imagePath'
        }
        
        # BPA measurement field mapping
        self.measurement_mapping = {
            'height': 'heightCm',
            'length': 'bodyLengthCm',
            'width': 'bodyWidthCm',
            'girth': 'chestGirthCm',
            'leg_length': 'legLengthCm',
            'head_length': 'headLengthCm',
            'tail_length': 'tailLengthCm',
            'bmi_estimate': 'bmiEstimate',
            'body_condition_score': 'bodyConditionScore',
            'symmetry_score': 'symmetryScore'
        }
        
        # BPA disease field mapping
        self.disease_mapping = {
            'name': 'diseaseName',
            'category': 'diseaseCategory',
            'confidence': 'detectionConfidence',
            'severity': 'severityLevel',
            'symptoms': 'symptoms',
            'treatment': 'recommendedTreatment',
            'description': 'diseaseDescription'
        }
    
    def map_analysis_to_bpa(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map AutoATC analysis data to BPA format.
        
        Args:
            analysis_data: AutoATC analysis data
            
        Returns:
            BPA formatted data
        """
        try:
            bpa_data = {
                'animalId': analysis_data.get('animal_id', ''),
                'analysisDate': self._format_datetime(analysis_data.get('analysis_date')),
                'atcScore': analysis_data.get('atc_score', 0.0),
                'atcGrade': analysis_data.get('atc_grade', 'N/A'),
                'breed': analysis_data.get('breed', 'Unknown'),
                'confidenceScore': analysis_data.get('confidence', 0.0),
                'imagePath': analysis_data.get('image_path', ''),
                'measurements': self._map_measurements(analysis_data.get('measurements', {})),
                'healthStatus': self._map_diseases(analysis_data.get('diseases', [])),
                'metadata': {
                    'source': 'AutoATC',
                    'version': '1.0',
                    'exportedAt': datetime.now().isoformat(),
                    'processingTime': analysis_data.get('processing_time', 0.0)
                }
            }
            
            return bpa_data
            
        except Exception as e:
            logger.error(f"Error mapping analysis to BPA: {e}")
            return {'error': str(e)}
    
    def _map_measurements(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """Map measurements to BPA format."""
        try:
            bpa_measurements = {}
            
            for atc_field, bpa_field in self.measurement_mapping.items():
                if atc_field in measurements and measurements[atc_field] is not None:
                    bpa_measurements[bpa_field] = measurements[atc_field]
            
            # Add units information
            bpa_measurements['units'] = 'cm'
            bpa_measurements['measurementDate'] = datetime.now().isoformat()
            
            return bpa_measurements
            
        except Exception as e:
            logger.warning(f"Error mapping measurements: {e}")
            return {}
    
    def _map_diseases(self, diseases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map diseases to BPA format."""
        try:
            bpa_diseases = []
            
            for disease in diseases:
                bpa_disease = {}
                
                for atc_field, bpa_field in self.disease_mapping.items():
                    if atc_field in disease:
                        bpa_disease[bpa_field] = disease[atc_field]
                
                # Add additional BPA fields
                bpa_disease['detectedAt'] = datetime.now().isoformat()
                bpa_disease['status'] = 'detected'
                
                bpa_diseases.append(bpa_disease)
            
            return bpa_diseases
            
        except Exception as e:
            logger.warning(f"Error mapping diseases: {e}")
            return []
    
    def _format_datetime(self, dt: Any) -> str:
        """Format datetime for BPA."""
        try:
            if isinstance(dt, str):
                return dt
            elif hasattr(dt, 'isoformat'):
                return dt.isoformat()
            else:
                return datetime.now().isoformat()
        except Exception as e:
            logger.warning(f"Error formatting datetime: {e}")
            return datetime.now().isoformat()
    
    def map_bpa_response(self, bpa_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map BPA response back to AutoATC format.
        
        Args:
            bpa_response: BPA API response
            
        Returns:
            AutoATC formatted response
        """
        try:
            atc_response = {
                'success': bpa_response.get('success', False),
                'message': bpa_response.get('message', ''),
                'bpa_animal_id': bpa_response.get('animalId', ''),
                'exported_at': bpa_response.get('exportedAt', datetime.now().isoformat()),
                'bpa_status': bpa_response.get('status', 'unknown')
            }
            
            return atc_response
            
        except Exception as e:
            logger.error(f"Error mapping BPA response: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_bpa_data(self, bpa_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate BPA data before sending.
        
        Args:
            bpa_data: BPA formatted data
            
        Returns:
            Validation results
        """
        try:
            validation = {
                'valid': True,
                'warnings': [],
                'errors': []
            }
            
            # Required fields
            required_fields = ['animalId', 'analysisDate', 'atcScore', 'atcGrade']
            for field in required_fields:
                if field not in bpa_data or bpa_data[field] is None:
                    validation['errors'].append(f'Missing required field: {field}')
                    validation['valid'] = False
            
            # Validate ATC score
            atc_score = bpa_data.get('atcScore', 0)
            if not isinstance(atc_score, (int, float)) or atc_score < 0 or atc_score > 100:
                validation['errors'].append('Invalid ATC score: must be between 0 and 100')
                validation['valid'] = False
            
            # Validate ATC grade
            atc_grade = bpa_data.get('atcGrade', '')
            valid_grades = ['A+', 'A', 'B+', 'B', 'C', 'D']
            if atc_grade not in valid_grades:
                validation['warnings'].append(f'Unusual ATC grade: {atc_grade}')
            
            # Validate measurements
            measurements = bpa_data.get('measurements', {})
            if measurements:
                for field, value in measurements.items():
                    if isinstance(value, (int, float)) and value < 0:
                        validation['warnings'].append(f'Negative measurement value: {field} = {value}')
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating BPA data: {e}")
            return {
                'valid': False,
                'warnings': [],
                'errors': [f'Validation error: {str(e)}']
            }
    
    def get_bpa_schema_info(self) -> Dict[str, Any]:
        """Get information about BPA schema mapping."""
        return {
            'field_mappings': self.bpa_field_mapping,
            'measurement_mappings': self.measurement_mapping,
            'disease_mappings': self.disease_mapping,
            'supported_fields': list(self.bpa_field_mapping.keys()),
            'version': '1.0'
        }

class ValidationSchemaMapper:
    """Maps validation data for accuracy assessment."""
    
    def __init__(self):
        """Initialize the validation schema mapper."""
        self.validation_metrics = [
            'atc_score_accuracy',
            'breed_classification_accuracy',
            'measurement_accuracy',
            'disease_detection_accuracy'
        ]
    
    def map_validation_data(self, ai_results: Dict[str, Any], 
                           manual_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map AI and manual results for validation.
        
        Args:
            ai_results: AI analysis results
            manual_results: Manual validation results
            
        Returns:
            Mapped validation data
        """
        try:
            validation_data = {
                'ai_results': {
                    'atc_score': ai_results.get('atc_score', 0.0),
                    'breed': ai_results.get('breed', 'unknown'),
                    'measurements': ai_results.get('measurements', {}),
                    'diseases': ai_results.get('diseases', []),
                    'confidence': ai_results.get('confidence', 0.0)
                },
                'manual_results': {
                    'atc_score': manual_results.get('manual_atc_score', 0.0),
                    'breed': manual_results.get('manual_breed', 'unknown'),
                    'measurements': manual_results.get('manual_measurements', {}),
                    'diseases': manual_results.get('manual_diseases', []),
                    'validator_notes': manual_results.get('validator_notes', '')
                },
                'comparison_metrics': self._calculate_comparison_metrics(ai_results, manual_results)
            }
            
            return validation_data
            
        except Exception as e:
            logger.error(f"Error mapping validation data: {e}")
            return {'error': str(e)}
    
    def _calculate_comparison_metrics(self, ai_results: Dict[str, Any], 
                                    manual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparison metrics between AI and manual results."""
        try:
            metrics = {}
            
            # ATC Score comparison
            ai_atc = ai_results.get('atc_score', 0.0)
            manual_atc = manual_results.get('manual_atc_score', 0.0)
            
            if manual_atc > 0:
                atc_difference = abs(ai_atc - manual_atc)
                atc_accuracy = max(0, 1 - (atc_difference / manual_atc))
                metrics['atc_score_difference'] = atc_difference
                metrics['atc_score_accuracy'] = atc_accuracy
            
            # Breed classification comparison
            ai_breed = ai_results.get('breed', 'unknown')
            manual_breed = manual_results.get('manual_breed', 'unknown')
            
            if manual_breed != 'unknown':
                breed_match = ai_breed.lower() == manual_breed.lower()
                metrics['breed_match'] = breed_match
                metrics['breed_classification_accuracy'] = 1.0 if breed_match else 0.0
            
            # Measurement comparison
            ai_measurements = ai_results.get('measurements', {})
            manual_measurements = manual_results.get('manual_measurements', {})
            
            if manual_measurements:
                measurement_accuracy = self._calculate_measurement_accuracy(
                    ai_measurements, manual_measurements
                )
                metrics['measurement_accuracy'] = measurement_accuracy
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating comparison metrics: {e}")
            return {}
    
    def _calculate_measurement_accuracy(self, ai_measurements: Dict[str, Any], 
                                      manual_measurements: Dict[str, Any]) -> float:
        """Calculate measurement accuracy between AI and manual results."""
        try:
            if not manual_measurements:
                return 0.0
            
            accuracies = []
            
            for measurement, manual_value in manual_measurements.items():
                if measurement in ai_measurements and manual_value > 0:
                    ai_value = ai_measurements[measurement]
                    if ai_value > 0:
                        # Calculate relative accuracy
                        relative_error = abs(ai_value - manual_value) / manual_value
                        accuracy = max(0, 1 - relative_error)
                        accuracies.append(accuracy)
            
            return np.mean(accuracies) if accuracies else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating measurement accuracy: {e}")
            return 0.0
    
    def generate_accuracy_report(self, validation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate accuracy report from validation records.
        
        Args:
            validation_records: List of validation records
            
        Returns:
            Accuracy report
        """
        try:
            if not validation_records:
                return {'error': 'No validation records provided'}
            
            # Calculate overall metrics
            atc_accuracies = []
            breed_accuracies = []
            measurement_accuracies = []
            confidences = []
            
            for record in validation_records:
                if 'atc_score_accuracy' in record:
                    atc_accuracies.append(record['atc_score_accuracy'])
                
                if 'breed_classification_accuracy' in record:
                    breed_accuracies.append(record['breed_classification_accuracy'])
                
                if 'measurement_accuracy' in record:
                    measurement_accuracies.append(record['measurement_accuracy'])
                
                if 'confidence' in record:
                    confidences.append(record['confidence'])
            
            # Calculate averages
            report = {
                'total_validations': len(validation_records),
                'atc_score_accuracy': np.mean(atc_accuracies) if atc_accuracies else 0.0,
                'breed_classification_accuracy': np.mean(breed_accuracies) if breed_accuracies else 0.0,
                'measurement_accuracy': np.mean(measurement_accuracies) if measurement_accuracies else 0.0,
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'validation_period': {
                    'start': min(record.get('created_at', '') for record in validation_records),
                    'end': max(record.get('created_at', '') for record in validation_records)
                },
                'recommendations': self._generate_recommendations(
                    atc_accuracies, breed_accuracies, measurement_accuracies
                )
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating accuracy report: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, atc_accuracies: List[float], 
                                breed_accuracies: List[float], 
                                measurement_accuracies: List[float]) -> List[str]:
        """Generate recommendations based on accuracy metrics."""
        recommendations = []
        
        # ATC Score recommendations
        if atc_accuracies:
            avg_atc_accuracy = np.mean(atc_accuracies)
            if avg_atc_accuracy < 0.7:
                recommendations.append("Improve ATC scoring algorithm - accuracy below 70%")
            elif avg_atc_accuracy < 0.8:
                recommendations.append("Fine-tune ATC scoring parameters for better accuracy")
        
        # Breed classification recommendations
        if breed_accuracies:
            avg_breed_accuracy = np.mean(breed_accuracies)
            if avg_breed_accuracy < 0.6:
                recommendations.append("Retrain breed classification model - accuracy below 60%")
            elif avg_breed_accuracy < 0.8:
                recommendations.append("Collect more training data for breed classification")
        
        # Measurement recommendations
        if measurement_accuracies:
            avg_measurement_accuracy = np.mean(measurement_accuracies)
            if avg_measurement_accuracy < 0.8:
                recommendations.append("Improve measurement calculation algorithms")
            elif avg_measurement_accuracy < 0.9:
                recommendations.append("Calibrate measurement scale factors")
        
        if not recommendations:
            recommendations.append("System performance is satisfactory across all metrics")
        
        return recommendations