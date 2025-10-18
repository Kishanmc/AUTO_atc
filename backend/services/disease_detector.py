"""
Disease Detection Service
Service layer for disease detection functionality
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add ai_modules to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_modules.disease.detector import DiseaseDetector

logger = logging.getLogger(__name__)

class DiseaseDetectionService:
    """Service for disease detection operations."""
    
    def __init__(self):
        """Initialize the disease detection service."""
        self.detector = DiseaseDetector()
        logger.info("Disease detection service initialized")
    
    def detect_diseases(self, image_data: bytes) -> Dict:
        """
        Detect diseases from image data.
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Dictionary containing disease detection results
        """
        try:
            import cv2
            import numpy as np
            from io import BytesIO
            from PIL import Image
            
            # Convert bytes to numpy array
            image = Image.open(BytesIO(image_data))
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Detect diseases
            result = self.detector.detect_diseases(image_array)
            
            # Add service metadata
            result['service'] = 'disease_detection'
            result['timestamp'] = self._get_timestamp()
            
            logger.info(f"Disease detection completed: {len(result.get('diseases', []))} diseases found")
            return result
            
        except Exception as e:
            logger.error(f"Error in disease detection service: {e}")
            return {
                'diseases': [],
                'confidence': 0.0,
                'service': 'disease_detection',
                'error': str(e)
            }
    
    def get_disease_information(self, disease: str) -> Dict:
        """
        Get detailed information about a specific disease.
        
        Args:
            disease: Disease identifier
            
        Returns:
            Dictionary containing disease information
        """
        try:
            return self.detector.get_disease_information(disease)
        except Exception as e:
            logger.error(f"Error getting disease information: {e}")
            return {'error': str(e)}
    
    def get_all_diseases(self) -> Dict:
        """
        Get information about all detectable diseases.
        
        Returns:
            Dictionary containing all disease information
        """
        try:
            return self.detector.get_all_diseases()
        except Exception as e:
            logger.error(f"Error getting all diseases: {e}")
            return {'error': str(e)}
    
    def validate_disease_detection(self, result: Dict) -> Dict:
        """
        Validate disease detection result.
        
        Args:
            result: Disease detection result
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation = {
                'valid': True,
                'confidence_level': 'low',
                'warnings': [],
                'recommendations': []
            }
            
            diseases = result.get('diseases', [])
            confidence = result.get('confidence', 0.0)
            
            # Check confidence level
            if confidence >= 0.8:
                validation['confidence_level'] = 'high'
            elif confidence >= 0.6:
                validation['confidence_level'] = 'medium'
            else:
                validation['confidence_level'] = 'low'
                validation['warnings'].append('Low confidence in disease detection')
            
            # Check if any diseases detected
            if not diseases:
                validation['warnings'].append('No diseases detected')
                validation['recommendations'].append('Animal appears healthy based on visual analysis')
            else:
                # Check severity of detected diseases
                severe_diseases = [d for d in diseases if d.get('severity', 'low') in ['high', 'critical']]
                if severe_diseases:
                    validation['warnings'].append(f'{len(severe_diseases)} severe diseases detected')
                    validation['recommendations'].append('Immediate veterinary consultation recommended')
                
                # Add specific recommendations for each disease
                for disease in diseases:
                    disease_name = disease.get('name', 'Unknown')
                    severity = disease.get('severity', 'low')
                    
                    if severity in ['high', 'critical']:
                        validation['recommendations'].append(f'Urgent attention needed for {disease_name}')
                    else:
                        validation['recommendations'].append(f'Monitor {disease_name} - consider veterinary checkup')
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating disease detection: {e}")
            return {
                'valid': False,
                'confidence_level': 'unknown',
                'warnings': [f'Validation error: {str(e)}'],
                'recommendations': []
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_service_status(self) -> Dict:
        """Get service status and capabilities."""
        return {
            'status': 'operational',
            'detector_loaded': self.detector.model is not None,
            'available_diseases': len(self.detector.disease_categories),
            'capabilities': {
                'deep_learning_detection': self.detector.model is not None,
                'rule_based_detection': True,
                'disease_information': True,
                'severity_assessment': True,
                'validation': True
            }
        }