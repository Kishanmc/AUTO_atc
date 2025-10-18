"""
Image processing utilities for AutoATC
"""

import cv2
import numpy as np
import base64
import io
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def process_image(image_data: bytes) -> np.ndarray:
    """
    Process uploaded image data.
    
    Args:
        image_data: Raw image data as bytes
        
    Returns:
        Processed image as numpy array
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        logger.info(f"Image processed successfully: {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise ValueError(f"Invalid image data: {str(e)}")

def save_image(image: np.ndarray, animal_id: str, filename: str) -> str:
    """
    Save processed image to disk.
    
    Args:
        image: Image as numpy array
        animal_id: Animal identifier
        filename: Original filename
        
    Returns:
        Path to saved image
    """
    try:
        # Create images directory if it doesn't exist
        images_dir = Path("static/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(filename).suffix or '.jpg'
        image_filename = f"{animal_id}_{timestamp}{file_extension}"
        image_path = images_dir / image_filename
        
        # Save image
        cv2.imwrite(str(image_path), image)
        
        # Return relative path
        relative_path = f"images/{image_filename}"
        logger.info(f"Image saved: {relative_path}")
        return relative_path
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise RuntimeError(f"Failed to save image: {str(e)}")

def generate_annotated_image(image: np.ndarray, 
                           detection_result: Dict,
                           keypoint_result: Optional[Dict] = None,
                           animal_id: str = None) -> str:
    """
    Generate annotated image with detection and keypoint results.
    
    Args:
        image: Original image
        detection_result: Detection results
        keypoint_result: Keypoint detection results
        animal_id: Animal identifier
        
    Returns:
        URL to annotated image
    """
    try:
        annotated_image = image.copy()
        
        # Draw detection bounding box
        if detection_result.get('detected', False) and detection_result.get('bounding_box'):
            bbox = detection_result['bounding_box']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            animal_type = detection_result.get('animal_type', 'unknown')
            confidence = detection_result.get('confidence', 0.0)
            label = f"{animal_type}: {confidence:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw keypoints
        if keypoint_result and keypoint_result.get('detected', False):
            keypoints = keypoint_result.get('keypoints', [])
            
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
                cv2.circle(annotated_image, (x, y), 3, color, -1)
                
                # Draw keypoint name for important points
                if kp['name'] in ['nose', 'left_eye', 'right_eye', 'left_shoulder', 'right_shoulder']:
                    cv2.putText(annotated_image, kp['name'], (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Save annotated image
        annotated_dir = Path("static/annotated")
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_filename = f"{animal_id}_annotated_{timestamp}.jpg"
        annotated_path = annotated_dir / annotated_filename
        
        cv2.imwrite(str(annotated_path), annotated_image)
        
        # Return URL path
        return f"annotated/{annotated_filename}"
        
    except Exception as e:
        logger.error(f"Error generating annotated image: {e}")
        return None

def resize_image(image: np.ndarray, max_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum size (width, height)
        
    Returns:
        Resized image
    """
    try:
        h, w = image.shape[:2]
        max_w, max_h = max_size
        
        # Calculate scaling factor
        scale = min(max_w / w, max_h / h)
        
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
        
    except Exception as e:
        logger.warning(f"Error resizing image: {e}")
        return image

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better analysis.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_image
        
    except Exception as e:
        logger.warning(f"Error enhancing image: {e}")
        return image

def validate_image_format(image_data: bytes) -> bool:
    """
    Validate image format and size.
    
    Args:
        image_data: Image data as bytes
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024
        if len(image_data) > max_size:
            logger.warning(f"Image too large: {len(image_data)} bytes")
            return False
        
        # Try to open with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Check format
        allowed_formats = ['JPEG', 'PNG', 'BMP', 'TIFF']
        if image.format not in allowed_formats:
            logger.warning(f"Unsupported format: {image.format}")
            return False
        
        # Check dimensions
        width, height = image.size
        if width < 100 or height < 100:
            logger.warning(f"Image too small: {width}x{height}")
            return False
        
        if width > 4000 or height > 4000:
            logger.warning(f"Image too large: {width}x{height}")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating image: {e}")
        return False

def extract_image_metadata(image: np.ndarray) -> Dict[str, Any]:
    """
    Extract metadata from image.
    
    Args:
        image: Input image
        
    Returns:
        Image metadata
    """
    try:
        h, w = image.shape[:2]
        
        metadata = {
            'width': w,
            'height': h,
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'aspect_ratio': w / h,
            'total_pixels': w * h,
            'color_space': 'BGR' if len(image.shape) == 3 else 'GRAY'
        }
        
        # Calculate basic statistics
        if len(image.shape) == 3:
            metadata['mean_brightness'] = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
            metadata['std_brightness'] = float(np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
        else:
            metadata['mean_brightness'] = float(np.mean(image))
            metadata['std_brightness'] = float(np.std(image))
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")
        return {}

def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Create thumbnail of image.
    
    Args:
        image: Input image
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    try:
        h, w = image.shape[:2]
        target_w, target_h = size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size if necessary
        if new_w != target_w or new_h != target_h:
            # Create black background
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Center the thumbnail
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail
            thumbnail = padded
        
        return thumbnail
        
    except Exception as e:
        logger.warning(f"Error creating thumbnail: {e}")
        return image

def encode_image_to_base64(image: np.ndarray, format: str = 'jpg') -> str:
    """
    Encode image to base64 string.
    
    Args:
        image: Input image
        format: Output format ('jpg', 'png')
        
    Returns:
        Base64 encoded string
    """
    try:
        # Encode image
        if format.lower() == 'jpg':
            _, buffer = cv2.imencode('.jpg', image)
        elif format.lower() == 'png':
            _, buffer = cv2.imencode('.png', image)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return ""

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to image.
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        Image as numpy array
    """
    try:
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
        
    except Exception as e:
        logger.error(f"Error decoding base64 to image: {e}")
        return np.array([])

def clean_old_images(max_age_days: int = 30):
    """
    Clean up old images from storage.
    
    Args:
        max_age_days: Maximum age of images in days
    """
    try:
        current_time = datetime.now()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        # Clean images directory
        images_dir = Path("static/images")
        if images_dir.exists():
            for file_path in images_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time.timestamp() - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        logger.info(f"Deleted old image: {file_path}")
        
        # Clean annotated images directory
        annotated_dir = Path("static/annotated")
        if annotated_dir.exists():
            for file_path in annotated_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time.timestamp() - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        logger.info(f"Deleted old annotated image: {file_path}")
        
        logger.info("Image cleanup completed")
        
    except Exception as e:
        logger.error(f"Error cleaning old images: {e}")

def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get information about an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image information
    """
    try:
        path = Path(image_path)
        
        if not path.exists():
            return {'error': 'File not found'}
        
        # Read image
        image = cv2.imread(str(path))
        if image is None:
            return {'error': 'Invalid image file'}
        
        # Get file info
        stat = path.stat()
        
        info = {
            'filename': path.name,
            'size_bytes': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'width': image.shape[1],
            'height': image.shape[0],
            'channels': image.shape[2] if len(image.shape) == 3 else 1
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        return {'error': str(e)}