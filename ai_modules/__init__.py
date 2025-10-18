"""
AutoATC AI Modules Package
Core AI functionality for animal type classification
"""

__version__ = "1.0.0"
__author__ = "AutoATC Team"

from .detection.animal_detector import AnimalDetector
from .detection.keypoint_detector import KeypointDetector
from .detection.aruco_detector import ArucoDetector
from .measurement.calculator import MeasurementCalculator
from .scoring.atc_scorer import ATCScorer
from .breed.classifier import BreedClassifier
from .disease.detector import DiseaseDetector

__all__ = [
    "AnimalDetector",
    "KeypointDetector", 
    "ArucoDetector",
    "MeasurementCalculator",
    "ATCScorer",
    "BreedClassifier",
    "DiseaseDetector"
]
