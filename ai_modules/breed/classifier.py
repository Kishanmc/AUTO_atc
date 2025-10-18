"""
Breed Classification Module
Classifies cattle and buffalo breeds using deep learning models
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class BreedClassifier:
    """Classifies cattle and buffalo breeds from images."""
    
    def __init__(self, model_path: str = "breed_classifier.pt", 
                 config_path: str = "breed_config.json"):
        """
        Initialize the breed classifier.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to breed configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Breed information
        self.cattle_breeds = {
            'gir': 'Gir',
            'sahival': 'Sahiwal', 
            'red_sindhi': 'Red Sindhi',
            'haryana': 'Haryana',
            'kankrej': 'Kankrej',
            'tharparkar': 'Tharparkar',
            'ongole': 'Ongole',
            'hallikar': 'Hallikar',
            'amritmahal': 'Amritmahal',
            'kangayam': 'Kangayam'
        }
        
        self.buffalo_breeds = {
            'murrah': 'Murrah',
            'nili_ravi': 'Nili-Ravi',
            'jafrabadi': 'Jafrabadi',
            'surti': 'Surti',
            'mehsana': 'Mehsana',
            'banni': 'Banni'
        }
        
        self.all_breeds = {**self.cattle_breeds, **self.buffalo_breeds}
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()
        self._load_config()
    
    def _load_model(self):
        """Load the trained breed classification model."""
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
                
                logger.info(f"Breed classification model loaded from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}, using rule-based classification")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error loading breed classification model: {e}")
            self.model = None
    
    def _create_model_architecture(self):
        """Create the model architecture."""
        # Simple CNN architecture for breed classification
        class BreedClassifierModel(nn.Module):
            def __init__(self, num_breeds):
                super(BreedClassifierModel, self).__init__()
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
                    nn.Linear(512, num_breeds)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return BreedClassifierModel(len(self.all_breeds))
    
    def _load_config(self):
        """Load breed configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._get_default_config()
                self._save_config()
        except Exception as e:
            logger.warning(f"Error loading breed config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """Get default breed configuration."""
        return {
            'breed_characteristics': {
                'gir': {
                    'color_patterns': ['brown', 'white', 'spotted'],
                    'horn_shape': 'curved',
                    'body_type': 'medium',
                    'distinctive_features': ['hump', 'drooping_ears']
                },
                'sahival': {
                    'color_patterns': ['red', 'brown'],
                    'horn_shape': 'straight',
                    'body_type': 'medium',
                    'distinctive_features': ['compact_body', 'short_horns']
                },
                'murrah': {
                    'color_patterns': ['black'],
                    'horn_shape': 'curved',
                    'body_type': 'large',
                    'distinctive_features': ['massive_body', 'short_horns']
                }
            },
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
        }
    
    def _save_config(self):
        """Save breed configuration."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving breed config: {e}")
    
    def classify_breed(self, image: np.ndarray, 
                      animal_type: Optional[str] = None) -> Dict:
        """
        Classify the breed of the animal in the image.
        
        Args:
            image: Input image as numpy array
            animal_type: Optional animal type ('cattle' or 'buffalo')
            
        Returns:
            Dictionary containing breed classification results
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            if self.model is not None:
                # Use deep learning model
                return self._classify_with_model(processed_image, animal_type)
            else:
                # Use rule-based classification
                return self._classify_with_rules(image, animal_type)
                
        except Exception as e:
            logger.error(f"Error in breed classification: {e}")
            return {
                'breed': 'unknown',
                'confidence': 0.0,
                'alternative_breeds': [],
                'method': 'error',
                'error': str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for classification."""
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image)
                return image_tensor.unsqueeze(0)  # Add batch dimension
            else:
                return image
                
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def _classify_with_model(self, image_tensor: torch.Tensor, 
                           animal_type: Optional[str]) -> Dict:
        """Classify breed using deep learning model."""
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probabilities, k=5)
                
                # Convert to breed names
                breed_names = list(self.all_breeds.keys())
                predictions = []
                
                for i in range(len(top_indices[0])):
                    idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    breed = breed_names[idx]
                    
                    # Filter by animal type if specified
                    if animal_type:
                        if animal_type == 'cattle' and breed in self.cattle_breeds:
                            predictions.append({'breed': breed, 'confidence': prob})
                        elif animal_type == 'buffalo' and breed in self.buffalo_breeds:
                            predictions.append({'breed': breed, 'confidence': prob})
                    else:
                        predictions.append({'breed': breed, 'confidence': prob})
                
                if predictions:
                    best_prediction = predictions[0]
                    alternative_breeds = predictions[1:4]  # Top 3 alternatives
                    
                    return {
                        'breed': best_prediction['breed'],
                        'confidence': best_prediction['confidence'],
                        'alternative_breeds': alternative_breeds,
                        'method': 'deep_learning',
                        'all_predictions': predictions
                    }
                else:
                    return {
                        'breed': 'unknown',
                        'confidence': 0.0,
                        'alternative_breeds': [],
                        'method': 'deep_learning',
                        'error': 'No valid predictions for specified animal type'
                    }
                    
        except Exception as e:
            logger.error(f"Error in model-based classification: {e}")
            return {
                'breed': 'unknown',
                'confidence': 0.0,
                'alternative_breeds': [],
                'method': 'deep_learning',
                'error': str(e)
            }
    
    def _classify_with_rules(self, image: np.ndarray, 
                           animal_type: Optional[str]) -> Dict:
        """Classify breed using rule-based approach."""
        try:
            # Extract visual features
            features = self._extract_visual_features(image)
            
            # Determine animal type if not specified
            if not animal_type:
                animal_type = self._determine_animal_type(features)
            
            # Get candidate breeds based on animal type
            if animal_type == 'cattle':
                candidate_breeds = list(self.cattle_breeds.keys())
            elif animal_type == 'buffalo':
                candidate_breeds = list(self.buffalo_breeds.keys())
            else:
                candidate_breeds = list(self.all_breeds.keys())
            
            # Score each breed based on features
            breed_scores = []
            for breed in candidate_breeds:
                score = self._score_breed_features(features, breed)
                breed_scores.append({'breed': breed, 'confidence': score})
            
            # Sort by confidence
            breed_scores.sort(key=lambda x: x['confidence'], reverse=True)
            
            if breed_scores and breed_scores[0]['confidence'] > 0.3:
                best_breed = breed_scores[0]
                alternative_breeds = breed_scores[1:4]
                
                return {
                    'breed': best_breed['breed'],
                    'confidence': best_breed['confidence'],
                    'alternative_breeds': alternative_breeds,
                    'method': 'rule_based',
                    'features_used': features
                }
            else:
                return {
                    'breed': 'unknown',
                    'confidence': 0.0,
                    'alternative_breeds': [],
                    'method': 'rule_based',
                    'error': 'Insufficient features for classification'
                }
                
        except Exception as e:
            logger.error(f"Error in rule-based classification: {e}")
            return {
                'breed': 'unknown',
                'confidence': 0.0,
                'alternative_breeds': [],
                'method': 'rule_based',
                'error': str(e)
            }
    
    def _extract_visual_features(self, image: np.ndarray) -> Dict:
        """Extract visual features from image for rule-based classification."""
        try:
            features = {}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Color analysis
            features['dominant_color'] = self._get_dominant_color(image)
            features['color_distribution'] = self._analyze_color_distribution(hsv)
            
            # Shape analysis
            features['body_shape'] = self._analyze_body_shape(image)
            features['horn_detection'] = self._detect_horns(image)
            
            # Texture analysis
            features['texture_pattern'] = self._analyze_texture(image)
            
            # Size estimation
            features['size_estimate'] = self._estimate_size(image)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {}
    
    def _get_dominant_color(self, image: np.ndarray) -> str:
        """Get dominant color in the image."""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges
            color_ranges = {
                'black': [(0, 0, 0), (180, 255, 50)],
                'brown': [(10, 50, 50), (20, 255, 255)],
                'red': [(0, 50, 50), (10, 255, 255)],
                'white': [(0, 0, 200), (180, 30, 255)],
                'gray': [(0, 0, 50), (180, 30, 200)]
            }
            
            max_pixels = 0
            dominant_color = 'unknown'
            
            for color, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = cv2.countNonZero(mask)
                
                if pixel_count > max_pixels:
                    max_pixels = pixel_count
                    dominant_color = color
            
            return dominant_color
            
        except Exception as e:
            logger.warning(f"Error getting dominant color: {e}")
            return 'unknown'
    
    def _analyze_color_distribution(self, hsv_image: np.ndarray) -> Dict:
        """Analyze color distribution in HSV space."""
        try:
            # Calculate histogram for each channel
            h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
            
            return {
                'hue_distribution': h_hist.flatten().tolist(),
                'saturation_distribution': s_hist.flatten().tolist(),
                'value_distribution': v_hist.flatten().tolist()
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing color distribution: {e}")
            return {}
    
    def _analyze_body_shape(self, image: np.ndarray) -> Dict:
        """Analyze body shape characteristics."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate shape features
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Circularity
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                return {
                    'area': area,
                    'perimeter': perimeter,
                    'aspect_ratio': aspect_ratio,
                    'circularity': circularity,
                    'width': w,
                    'height': h
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Error analyzing body shape: {e}")
            return {}
    
    def _detect_horns(self, image: np.ndarray) -> Dict:
        """Detect and analyze horn characteristics."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines (potential horns)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=30, maxLineGap=10)
            
            horn_features = {
                'detected': False,
                'count': 0,
                'length': 0,
                'angle': 0
            }
            
            if lines is not None and len(lines) > 0:
                # Analyze line characteristics
                line_lengths = []
                line_angles = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    line_lengths.append(length)
                    line_angles.append(angle)
                
                # Filter for potential horns (longer lines)
                long_lines = [l for l in line_lengths if l > 50]
                
                if long_lines:
                    horn_features['detected'] = True
                    horn_features['count'] = len(long_lines)
                    horn_features['length'] = np.mean(long_lines)
                    horn_features['angle'] = np.mean(line_angles)
            
            return horn_features
            
        except Exception as e:
            logger.warning(f"Error detecting horns: {e}")
            return {'detected': False, 'count': 0, 'length': 0, 'angle': 0}
    
    def _analyze_texture(self, image: np.ndarray) -> Dict:
        """Analyze texture patterns in the image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture features using Local Binary Pattern
            from skimage.feature import local_binary_pattern
            
            # LBP parameters
            radius = 1
            n_points = 8 * radius
            
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate texture statistics
            texture_features = {
                'mean': float(np.mean(lbp)),
                'std': float(np.std(lbp)),
                'energy': float(np.sum(lbp**2)),
                'entropy': float(-np.sum(lbp * np.log2(lbp + 1e-10)))
            }
            
            return texture_features
            
        except Exception as e:
            logger.warning(f"Error analyzing texture: {e}")
            return {}
    
    def _estimate_size(self, image: np.ndarray) -> Dict:
        """Estimate animal size from image."""
        try:
            height, width = image.shape[:2]
            
            # Simple size estimation based on image dimensions
            # This would need calibration with real measurements
            estimated_size = {
                'image_height': height,
                'image_width': width,
                'estimated_animal_height': height * 0.8,  # Rough estimate
                'estimated_animal_width': width * 0.6
            }
            
            return estimated_size
            
        except Exception as e:
            logger.warning(f"Error estimating size: {e}")
            return {}
    
    def _determine_animal_type(self, features: Dict) -> str:
        """Determine animal type (cattle vs buffalo) from features."""
        try:
            # Simple rule-based classification
            dominant_color = features.get('dominant_color', 'unknown')
            body_shape = features.get('body_shape', {})
            horn_features = features.get('horn_detection', {})
            
            # Buffalo characteristics (typically black, larger body)
            if dominant_color == 'black' and body_shape.get('area', 0) > 10000:
                return 'buffalo'
            
            # Cattle characteristics (more varied colors, smaller)
            return 'cattle'
            
        except Exception as e:
            logger.warning(f"Error determining animal type: {e}")
            return 'cattle'
    
    def _score_breed_features(self, features: Dict, breed: str) -> float:
        """Score breed based on extracted features."""
        try:
            if breed not in self.config['breed_characteristics']:
                return 0.0
            
            breed_chars = self.config['breed_characteristics'][breed]
            score = 0.0
            total_weight = 0.0
            
            # Color matching
            dominant_color = features.get('dominant_color', 'unknown')
            expected_colors = breed_chars.get('color_patterns', [])
            
            if dominant_color in expected_colors:
                score += 0.4
            total_weight += 0.4
            
            # Body type matching
            body_shape = features.get('body_shape', {})
            area = body_shape.get('area', 0)
            expected_body_type = breed_chars.get('body_type', 'medium')
            
            if expected_body_type == 'large' and area > 15000:
                score += 0.3
            elif expected_body_type == 'medium' and 8000 < area < 15000:
                score += 0.3
            elif expected_body_type == 'small' and area < 8000:
                score += 0.3
            total_weight += 0.3
            
            # Horn characteristics
            horn_features = features.get('horn_detection', {})
            if horn_features.get('detected', False):
                expected_horn_shape = breed_chars.get('horn_shape', 'unknown')
                # Simple horn shape matching (would need more sophisticated analysis)
                score += 0.2
            total_weight += 0.2
            
            # Additional features
            distinctive_features = breed_chars.get('distinctive_features', [])
            # This would need more sophisticated feature detection
            score += 0.1
            total_weight += 0.1
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error scoring breed features: {e}")
            return 0.0
    
    def get_breed_information(self, breed: str) -> Dict:
        """Get detailed information about a specific breed."""
        try:
            if breed in self.all_breeds:
                breed_info = {
                    'breed_id': breed,
                    'breed_name': self.all_breeds[breed],
                    'animal_type': 'cattle' if breed in self.cattle_breeds else 'buffalo',
                    'characteristics': self.config['breed_characteristics'].get(breed, {}),
                    'description': self._get_breed_description(breed)
                }
                return breed_info
            else:
                return {'error': 'Breed not found'}
                
        except Exception as e:
            logger.error(f"Error getting breed information: {e}")
            return {'error': str(e)}
    
    def _get_breed_description(self, breed: str) -> str:
        """Get description for a breed."""
        descriptions = {
            'gir': 'Gir cattle are known for their distinctive hump and drooping ears. They are excellent milk producers.',
            'sahival': 'Sahiwal cattle are known for their heat tolerance and good milk production in tropical climates.',
            'murrah': 'Murrah buffaloes are the best milk producers among buffalo breeds, known for their massive body.',
            'nili_ravi': 'Nili-Ravi buffaloes are known for their dual-purpose nature and good milk production.',
            'jafrabadi': 'Jafrabadi buffaloes are known for their large size and good milk production.',
            'surti': 'Surti buffaloes are known for their adaptability and good milk production.',
            'mehsana': 'Mehsana buffaloes are known for their good milk production and adaptability.',
            'banni': 'Banni buffaloes are known for their hardiness and good milk production in arid regions.'
        }
        
        return descriptions.get(breed, 'No description available.')
    
    def get_all_breeds(self) -> Dict:
        """Get information about all available breeds."""
        return {
            'cattle_breeds': self.cattle_breeds,
            'buffalo_breeds': self.buffalo_breeds,
            'total_breeds': len(self.all_breeds)
        }
