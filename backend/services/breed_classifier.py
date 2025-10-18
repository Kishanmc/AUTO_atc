"""
Breed Classification Service
Service layer for breed classification functionality
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import logging

# Add ai_modules to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_modules.breed.classifier import BreedClassifier

logger = logging.getLogger(__name__)

class BreedClassificationService:
    """Service for breed classification operations."""
    
    def __init__(self):
        """Initialize the breed classification service."""
        self.classifier = BreedClassifier()
        logger.info("Breed classification service initialized")
    
    def classify_breed(self, image_data: bytes, 
                      animal_type: Optional[str] = None) -> Dict:
        """
        Classify breed from image data.
        
        Args:
            image_data: Image data as bytes
            animal_type: Optional animal type ('cattle' or 'buffalo')
            
        Returns:
            Dictionary containing breed classification results
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
            
            # Classify breed
            result = self.classifier.classify_breed(image_array, animal_type)
            
            # Add service metadata
            result['service'] = 'breed_classification'
            result['timestamp'] = self._get_timestamp()
            
            logger.info(f"Breed classification completed: {result.get('breed', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error in breed classification service: {e}")
            return {
                'breed': 'unknown',
                'confidence': 0.0,
                'alternative_breeds': [],
                'method': 'error',
                'service': 'breed_classification',
                'error': str(e)
            }
    
    def get_breed_information(self, breed: str) -> Dict:
        """
        Get detailed information about a specific breed.
        
        Args:
            breed: Breed identifier
            
        Returns:
            Dictionary containing breed information
        """
        try:
            return self.classifier.get_breed_information(breed)
        except Exception as e:
            logger.error(f"Error getting breed information: {e}")
            return {'error': str(e)}
    
    def get_all_breeds(self) -> Dict:
        """
        Get information about all available breeds.
        
        Returns:
            Dictionary containing all breed information
        """
        try:
            return self.classifier.get_all_breeds()
        except Exception as e:
            logger.error(f"Error getting all breeds: {e}")
            return {'error': str(e)}
    
    def validate_breed_classification(self, result: Dict) -> Dict:
        """
        Validate breed classification result.
        
        Args:
            result: Breed classification result
            
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
            
            confidence = result.get('confidence', 0.0)
            breed = result.get('breed', 'unknown')
            method = result.get('method', 'unknown')
            
            # Check confidence level
            if confidence >= 0.8:
                validation['confidence_level'] = 'high'
            elif confidence >= 0.6:
                validation['confidence_level'] = 'medium'
            else:
                validation['confidence_level'] = 'low'
                validation['warnings'].append('Low confidence in breed classification')
            
            # Check if breed is unknown
            if breed == 'unknown':
                validation['valid'] = False
                validation['warnings'].append('Unable to classify breed')
                validation['recommendations'].append('Try with a clearer image or different angle')
            
            # Check method used
            if method == 'rule_based':
                validation['warnings'].append('Using rule-based classification (less accurate)')
                validation['recommendations'].append('Consider training a deep learning model for better accuracy')
            
            # Add recommendations based on confidence
            if confidence < 0.5:
                validation['recommendations'].extend([
                    'Ensure animal is clearly visible in image',
                    'Try different lighting conditions',
                    'Ensure animal is in proper pose'
                ])
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating breed classification: {e}")
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
            'classifier_loaded': self.classifier.model is not None,
            'available_breeds': len(self.classifier.all_breeds),
            'cattle_breeds': len(self.classifier.cattle_breeds),
            'buffalo_breeds': len(self.classifier.buffalo_breeds),
            'capabilities': {
                'deep_learning_classification': self.classifier.model is not None,
                'rule_based_classification': True,
                'breed_information': True,
                'validation': True
            }
        }