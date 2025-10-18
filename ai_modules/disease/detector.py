"""
Disease Detection Module
Detects common diseases and health issues in cattle and buffaloes
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DiseaseDetector:
    """Detects diseases and health issues in cattle and buffaloes."""
    
    def __init__(self, model_path: str = "disease_detector.pt", 
                 config_path: str = "disease_config.json"):
        """
        Initialize the disease detector.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to disease configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Disease categories and information
        self.disease_categories = {
            'skin_conditions': {
                'mange': {
                    'name': 'Mange',
                    'severity': 'medium',
                    'symptoms': ['hair_loss', 'scaly_skin', 'itching'],
                    'treatment': 'Acaricide treatment',
                    'description': 'Parasitic skin condition caused by mites'
                },
                'ringworm': {
                    'name': 'Ringworm',
                    'severity': 'low',
                    'symptoms': ['circular_lesions', 'hair_loss', 'scaly_patches'],
                    'treatment': 'Antifungal treatment',
                    'description': 'Fungal skin infection'
                },
                'dermatitis': {
                    'name': 'Dermatitis',
                    'severity': 'medium',
                    'symptoms': ['inflammation', 'redness', 'swelling'],
                    'treatment': 'Anti-inflammatory treatment',
                    'description': 'Skin inflammation'
                }
            },
            'hoof_conditions': {
                'foot_rot': {
                    'name': 'Foot Rot',
                    'severity': 'high',
                    'symptoms': ['lameness', 'swelling', 'foul_smell'],
                    'treatment': 'Antibiotic treatment and hoof trimming',
                    'description': 'Bacterial infection of the hoof'
                },
                'laminitis': {
                    'name': 'Laminitis',
                    'severity': 'high',
                    'symptoms': ['lameness', 'warm_hooves', 'reluctance_to_move'],
                    'treatment': 'Anti-inflammatory treatment and rest',
                    'description': 'Inflammation of the hoof laminae'
                }
            },
            'eye_conditions': {
                'conjunctivitis': {
                    'name': 'Conjunctivitis',
                    'severity': 'medium',
                    'symptoms': ['red_eyes', 'discharge', 'swelling'],
                    'treatment': 'Antibiotic eye drops',
                    'description': 'Inflammation of the eye membrane'
                },
                'keratitis': {
                    'name': 'Keratitis',
                    'severity': 'high',
                    'symptoms': ['cloudy_eyes', 'sensitivity_to_light', 'discharge'],
                    'treatment': 'Veterinary treatment required',
                    'description': 'Inflammation of the cornea'
                }
            },
            'general_health': {
                'malnutrition': {
                    'name': 'Malnutrition',
                    'severity': 'medium',
                    'symptoms': ['weight_loss', 'poor_condition', 'weakness'],
                    'treatment': 'Improved nutrition and veterinary consultation',
                    'description': 'Inadequate nutrition leading to poor health'
                },
                'dehydration': {
                    'name': 'Dehydration',
                    'severity': 'high',
                    'symptoms': ['sunken_eyes', 'dry_mucous_membranes', 'lethargy'],
                    'treatment': 'Fluid therapy and veterinary attention',
                    'description': 'Insufficient water intake'
                }
            }
        }
        
        self._load_model()
        self._load_config()
    
    def _load_model(self):
        """Load the trained disease detection model."""
        try:
            if Path(self.model_path).exists():
                # Load model checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Initialize model architecture
                self.model = self._create_model_architecture()
                
                # Load model weights
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Disease detection model loaded from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}, using rule-based detection")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error loading disease detection model: {e}")
            self.model = None
    
    def _create_model_architecture(self):
        """Create the model architecture for disease detection."""
        class DiseaseDetectionModel(nn.Module):
            def __init__(self, num_diseases):
                super(DiseaseDetectionModel, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(256 * 7 * 7, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_diseases)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        # Count total diseases
        total_diseases = sum(len(category) for category in self.disease_categories.values())
        return DiseaseDetectionModel(total_diseases)
    
    def _load_config(self):
        """Load disease configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._get_default_config()
                self._save_config()
        except Exception as e:
            logger.warning(f"Error loading disease config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """Get default disease configuration."""
        return {
            'detection_thresholds': {
                'high_confidence': 0.8,
                'medium_confidence': 0.6,
                'low_confidence': 0.4
            },
            'severity_weights': {
                'critical': 1.0,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            },
            'visual_features': {
                'skin_conditions': ['color_changes', 'texture_changes', 'lesions'],
                'hoof_conditions': ['swelling', 'discoloration', 'deformities'],
                'eye_conditions': ['discharge', 'redness', 'cloudiness'],
                'general_health': ['body_condition', 'posture', 'alertness']
            }
        }
    
    def _save_config(self):
        """Save disease configuration."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving disease config: {e}")
    
    def detect_diseases(self, image: np.ndarray) -> Dict:
        """
        Detect diseases in the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing disease detection results
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            if self.model is not None:
                # Use deep learning model
                return self._detect_with_model(processed_image)
            else:
                # Use rule-based detection
                return self._detect_with_rules(image)
                
        except Exception as e:
            logger.error(f"Error in disease detection: {e}")
            return {
                'diseases': [],
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for disease detection."""
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (224, 224))
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            # Convert to tensor
            import torch
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def _detect_with_model(self, image_tensor: torch.Tensor) -> Dict:
        """Detect diseases using deep learning model."""
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                probabilities = torch.sigmoid(outputs)
                
                # Get disease names
                disease_names = self._get_disease_names()
                
                # Filter diseases with confidence above threshold
                threshold = self.config['detection_thresholds']['low_confidence']
                detected_diseases = []
                
                for i, prob in enumerate(probabilities[0]):
                    if prob.item() > threshold and i < len(disease_names):
                        disease_info = self._get_disease_info(disease_names[i])
                        if disease_info:
                            disease_info['confidence'] = float(prob.item())
                            detected_diseases.append(disease_info)
                
                # Sort by confidence
                detected_diseases.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Calculate overall confidence
                overall_confidence = max([d['confidence'] for d in detected_diseases]) if detected_diseases else 0.0
                
                return {
                    'diseases': detected_diseases,
                    'confidence': overall_confidence,
                    'method': 'deep_learning',
                    'total_diseases_found': len(detected_diseases)
                }
                
        except Exception as e:
            logger.error(f"Error in model-based detection: {e}")
            return {
                'diseases': [],
                'confidence': 0.0,
                'method': 'deep_learning',
                'error': str(e)
            }
    
    def _detect_with_rules(self, image: np.ndarray) -> Dict:
        """Detect diseases using rule-based approach."""
        try:
            detected_diseases = []
            
            # Analyze different body regions
            regions = self._extract_body_regions(image)
            
            # Check for skin conditions
            skin_diseases = self._check_skin_conditions(regions.get('skin', image))
            detected_diseases.extend(skin_diseases)
            
            # Check for hoof conditions
            hoof_diseases = self._check_hoof_conditions(regions.get('hooves', image))
            detected_diseases.extend(hoof_diseases)
            
            # Check for eye conditions
            eye_diseases = self._check_eye_conditions(regions.get('eyes', image))
            detected_diseases.extend(eye_diseases)
            
            # Check for general health indicators
            general_diseases = self._check_general_health(image)
            detected_diseases.extend(general_diseases)
            
            # Calculate overall confidence
            overall_confidence = max([d['confidence'] for d in detected_diseases]) if detected_diseases else 0.0
            
            return {
                'diseases': detected_diseases,
                'confidence': overall_confidence,
                'method': 'rule_based',
                'total_diseases_found': len(detected_diseases)
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based detection: {e}")
            return {
                'diseases': [],
                'confidence': 0.0,
                'method': 'rule_based',
                'error': str(e)
            }
    
    def _extract_body_regions(self, image: np.ndarray) -> Dict:
        """Extract different body regions for analysis."""
        try:
            h, w = image.shape[:2]
            
            regions = {
                'skin': image[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)],  # Main body
                'eyes': image[int(h*0.1):int(h*0.3), int(w*0.2):int(w*0.8)],  # Head area
                'hooves': image[int(h*0.7):h, :],  # Lower body
                'full': image
            }
            
            return regions
            
        except Exception as e:
            logger.warning(f"Error extracting body regions: {e}")
            return {'full': image}
    
    def _check_skin_conditions(self, skin_region: np.ndarray) -> List[Dict]:
        """Check for skin-related diseases."""
        try:
            diseases = []
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
            
            # Check for color abnormalities
            color_abnormalities = self._detect_color_abnormalities(hsv)
            if color_abnormalities > 0.3:
                diseases.append({
                    'name': 'Dermatitis',
                    'category': 'skin_conditions',
                    'confidence': min(color_abnormalities, 0.8),
                    'severity': 'medium',
                    'symptoms': ['inflammation', 'redness'],
                    'treatment': 'Anti-inflammatory treatment',
                    'description': 'Skin inflammation detected'
                })
            
            # Check for texture changes
            texture_changes = self._detect_texture_changes(gray)
            if texture_changes > 0.4:
                diseases.append({
                    'name': 'Mange',
                    'category': 'skin_conditions',
                    'confidence': min(texture_changes, 0.7),
                    'severity': 'medium',
                    'symptoms': ['scaly_skin', 'hair_loss'],
                    'treatment': 'Acaricide treatment',
                    'description': 'Parasitic skin condition'
                })
            
            return diseases
            
        except Exception as e:
            logger.warning(f"Error checking skin conditions: {e}")
            return []
    
    def _check_hoof_conditions(self, hoof_region: np.ndarray) -> List[Dict]:
        """Check for hoof-related diseases."""
        try:
            diseases = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(hoof_region, cv2.COLOR_BGR2GRAY)
            
            # Check for swelling
            swelling = self._detect_swelling(gray)
            if swelling > 0.5:
                diseases.append({
                    'name': 'Foot Rot',
                    'category': 'hoof_conditions',
                    'confidence': min(swelling, 0.8),
                    'severity': 'high',
                    'symptoms': ['swelling', 'lameness'],
                    'treatment': 'Antibiotic treatment and hoof trimming',
                    'description': 'Bacterial hoof infection'
                })
            
            return diseases
            
        except Exception as e:
            logger.warning(f"Error checking hoof conditions: {e}")
            return []
    
    def _check_eye_conditions(self, eye_region: np.ndarray) -> List[Dict]:
        """Check for eye-related diseases."""
        try:
            diseases = []
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
            
            # Check for redness
            redness = self._detect_redness(hsv)
            if redness > 0.4:
                diseases.append({
                    'name': 'Conjunctivitis',
                    'category': 'eye_conditions',
                    'confidence': min(redness, 0.7),
                    'severity': 'medium',
                    'symptoms': ['red_eyes', 'discharge'],
                    'treatment': 'Antibiotic eye drops',
                    'description': 'Eye inflammation'
                })
            
            return diseases
            
        except Exception as e:
            logger.warning(f"Error checking eye conditions: {e}")
            return []
    
    def _check_general_health(self, image: np.ndarray) -> List[Dict]:
        """Check for general health indicators."""
        try:
            diseases = []
            
            # Analyze overall body condition
            body_condition = self._assess_body_condition(image)
            
            if body_condition < 0.3:  # Poor body condition
                diseases.append({
                    'name': 'Malnutrition',
                    'category': 'general_health',
                    'confidence': 0.6,
                    'severity': 'medium',
                    'symptoms': ['poor_condition', 'weight_loss'],
                    'treatment': 'Improved nutrition and veterinary consultation',
                    'description': 'Inadequate nutrition'
                })
            
            return diseases
            
        except Exception as e:
            logger.warning(f"Error checking general health: {e}")
            return []
    
    def _detect_color_abnormalities(self, hsv_image: np.ndarray) -> float:
        """Detect color abnormalities in HSV image."""
        try:
            # Define normal skin color range
            normal_hue_range = (10, 30)  # Yellowish to reddish
            normal_sat_range = (50, 255)
            normal_val_range = (100, 255)
            
            # Create mask for normal skin color
            mask = cv2.inRange(hsv_image, 
                             np.array([normal_hue_range[0], normal_sat_range[0], normal_val_range[0]]),
                             np.array([normal_hue_range[1], normal_sat_range[1], normal_val_range[1]]))
            
            # Calculate percentage of abnormal colors
            total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
            abnormal_pixels = total_pixels - cv2.countNonZero(mask)
            abnormality_ratio = abnormal_pixels / total_pixels
            
            return abnormality_ratio
            
        except Exception as e:
            logger.warning(f"Error detecting color abnormalities: {e}")
            return 0.0
    
    def _detect_texture_changes(self, gray_image: np.ndarray) -> float:
        """Detect texture changes in grayscale image."""
        try:
            # Calculate local binary pattern
            from skimage.feature import local_binary_pattern
            
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            
            # Calculate texture variance
            texture_variance = np.var(lbp)
            
            # Normalize to 0-1 range
            normalized_variance = min(texture_variance / 1000, 1.0)
            
            return normalized_variance
            
        except Exception as e:
            logger.warning(f"Error detecting texture changes: {e}")
            return 0.0
    
    def _detect_swelling(self, gray_image: np.ndarray) -> float:
        """Detect swelling in grayscale image."""
        try:
            # Edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Calculate contour area
                total_area = sum(cv2.contourArea(c) for c in contours)
                image_area = gray_image.shape[0] * gray_image.shape[1]
                
                # Swelling indicated by high contour density
                swelling_ratio = total_area / image_area
                return min(swelling_ratio * 10, 1.0)  # Scale to 0-1
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error detecting swelling: {e}")
            return 0.0
    
    def _detect_redness(self, hsv_image: np.ndarray) -> float:
        """Detect redness in HSV image."""
        try:
            # Define red color range in HSV
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            
            # Create masks for red color
            mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
            mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
            red_mask = mask1 + mask2
            
            # Calculate redness ratio
            total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
            red_pixels = cv2.countNonZero(red_mask)
            redness_ratio = red_pixels / total_pixels
            
            return min(redness_ratio * 5, 1.0)  # Scale to 0-1
            
        except Exception as e:
            logger.warning(f"Error detecting redness: {e}")
            return 0.0
    
    def _assess_body_condition(self, image: np.ndarray) -> float:
        """Assess overall body condition."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (likely the animal)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate area and perimeter
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Calculate circularity (indicator of body condition)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    return min(circularity, 1.0)
            
            return 0.5  # Default moderate condition
            
        except Exception as e:
            logger.warning(f"Error assessing body condition: {e}")
            return 0.5
    
    def _get_disease_names(self) -> List[str]:
        """Get list of all disease names."""
        disease_names = []
        for category in self.disease_categories.values():
            for disease_id in category.keys():
                disease_names.append(disease_id)
        return disease_names
    
    def _get_disease_info(self, disease_id: str) -> Optional[Dict]:
        """Get disease information by ID."""
        for category in self.disease_categories.values():
            if disease_id in category:
                disease_info = category[disease_id].copy()
                disease_info['disease_id'] = disease_id
                return disease_info
        return None
    
    def get_disease_information(self, disease: str) -> Dict:
        """Get detailed information about a specific disease."""
        try:
            disease_info = self._get_disease_info(disease)
            if disease_info:
                return disease_info
            else:
                return {'error': 'Disease not found'}
                
        except Exception as e:
            logger.error(f"Error getting disease information: {e}")
            return {'error': str(e)}
    
    def get_all_diseases(self) -> Dict:
        """Get information about all detectable diseases."""
        try:
            all_diseases = {}
            for category_name, diseases in self.disease_categories.items():
                all_diseases[category_name] = diseases
            
            return {
                'disease_categories': all_diseases,
                'total_diseases': sum(len(diseases) for diseases in self.disease_categories.values()),
                'categories': list(self.disease_categories.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting all diseases: {e}")
            return {'error': str(e)}