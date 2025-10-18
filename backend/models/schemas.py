"""
Pydantic schemas for AutoATC API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enums
class AnimalType(str, Enum):
    CATTLE = "cattle"
    BUFFALO = "buffalo"
    UNKNOWN = "unknown"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Base schemas
class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")

class KeypointData(BaseModel):
    id: int = Field(..., description="Keypoint ID")
    name: str = Field(..., description="Keypoint name")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(0.0, description="Z coordinate")
    visibility: float = Field(..., ge=0.0, le=1.0, description="Visibility score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")

class MeasurementData(BaseModel):
    height: Optional[float] = Field(None, description="Height in cm")
    length: Optional[float] = Field(None, description="Length in cm")
    width: Optional[float] = Field(None, description="Width in cm")
    girth: Optional[float] = Field(None, description="Chest girth in cm")
    leg_length: Optional[float] = Field(None, description="Leg length in cm")
    head_length: Optional[float] = Field(None, description="Head length in cm")
    tail_length: Optional[float] = Field(None, description="Tail length in cm")
    bmi_estimate: Optional[float] = Field(None, description="Body mass index estimate")
    body_condition_score: Optional[float] = Field(None, ge=1.0, le=5.0, description="Body condition score (1-5)")
    symmetry_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Body symmetry score")

class ATCScore(BaseModel):
    score: float = Field(..., ge=0.0, le=100.0, description="ATC score (0-100)")
    grade: str = Field(..., description="ATC grade (A+, A, B+, B, C, D)")
    factors: Dict[str, float] = Field(..., description="Individual scoring factors")
    recommendations: List[str] = Field(..., description="Improvement recommendations")

class BreedClassification(BaseModel):
    breed: Optional[str] = Field(None, description="Predicted breed")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Classification confidence")
    alternative_breeds: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative breed predictions")

class DiseaseDetection(BaseModel):
    name: str = Field(..., description="Disease name")
    category: str = Field(..., description="Disease category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    severity: SeverityLevel = Field(..., description="Disease severity")
    symptoms: List[str] = Field(..., description="Observed symptoms")
    treatment: str = Field(..., description="Recommended treatment")
    description: str = Field(..., description="Disease description")

# Request schemas
class AnalyzeRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    filename: str = Field(..., description="Original filename")
    animal_id: Optional[str] = Field(None, description="Optional animal ID")
    include_breed_classification: bool = Field(True, description="Include breed classification")
    include_disease_detection: bool = Field(True, description="Include disease detection")
    include_measurements: bool = Field(True, description="Include body measurements")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v or len(v) < 100:  # Basic validation
            raise ValueError('Invalid image data')
        return v

class ExportBPARequest(BaseModel):
    animal_id: str = Field(..., description="Animal ID to export")
    bpa_api_key: Optional[str] = Field(None, description="BPA API key")
    include_measurements: bool = Field(True, description="Include measurements in export")
    include_diseases: bool = Field(True, description="Include disease information in export")

class ValidationRequest(BaseModel):
    animal_id: str = Field(..., description="Animal ID to validate")
    manual_atc_score: float = Field(..., ge=0.0, le=100.0, description="Manual ATC score")
    manual_breed: Optional[str] = Field(None, description="Manual breed classification")
    manual_measurements: Optional[Dict[str, float]] = Field(None, description="Manual measurements")
    validator_notes: Optional[str] = Field(None, description="Validator notes")

# Response schemas
class AnalysisResult(BaseModel):
    animal_id: str = Field(..., description="Animal ID")
    animal_type: AnimalType = Field(..., description="Detected animal type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    bounding_box: Optional[BoundingBox] = Field(None, description="Animal bounding box")
    keypoints: List[KeypointData] = Field(default_factory=list, description="Detected keypoints")
    measurements: Optional[MeasurementData] = Field(None, description="Body measurements")
    atc_score: Optional[ATCScore] = Field(None, description="ATC scoring results")
    breed_classification: Optional[BreedClassification] = Field(None, description="Breed classification")
    diseases: List[DiseaseDetection] = Field(default_factory=list, description="Detected diseases")
    processing_time: float = Field(..., description="Processing time in seconds")
    analysis_date: datetime = Field(..., description="Analysis timestamp")
    status: AnalysisStatus = Field(..., description="Analysis status")

class AnalyzeResponse(BaseModel):
    data: AnalysisResult = Field(..., description="Analysis results")
    annotated_image_url: Optional[str] = Field(None, description="URL to annotated image")
    message: str = Field(..., description="Response message")

class ResultsResponse(BaseModel):
    animal_id: str = Field(..., description="Animal ID")
    analysis: AnalysisResult = Field(..., description="Analysis results")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

class ExportResponse(BaseModel):
    success: bool = Field(..., description="Export success status")
    message: str = Field(..., description="Export message")
    bpa_animal_id: Optional[str] = Field(None, description="BPA assigned animal ID")
    exported_data: Optional[Dict[str, Any]] = Field(None, description="Exported data")

class ValidationResponse(BaseModel):
    animal_id: str = Field(..., description="Animal ID")
    validation_id: int = Field(..., description="Validation record ID")
    accuracy_metrics: Dict[str, float] = Field(..., description="Accuracy metrics")
    atc_score_difference: float = Field(..., description="ATC score difference")
    breed_match: Optional[bool] = Field(None, description="Breed classification match")
    measurement_accuracy: Optional[float] = Field(None, description="Measurement accuracy")
    validation_date: datetime = Field(..., description="Validation timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

# Status and health check schemas
class HealthCheck(BaseModel):
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")

class APIStatus(BaseModel):
    api_status: str = Field(..., description="API status")
    database: str = Field(..., description="Database status")
    ai_modules: Dict[str, str] = Field(..., description="AI modules status")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")

# BPA integration schemas
class BPAAnalysisData(BaseModel):
    animal_id: str = Field(..., description="Animal ID")
    analysis_date: datetime = Field(..., description="Analysis date")
    atc_score: float = Field(..., description="ATC score")
    atc_grade: str = Field(..., description="ATC grade")
    breed: Optional[str] = Field(None, description="Breed classification")
    measurements: Optional[Dict[str, float]] = Field(None, description="Body measurements")
    diseases: Optional[List[Dict[str, Any]]] = Field(None, description="Detected diseases")
    confidence: float = Field(..., description="Overall confidence")
    image_path: str = Field(..., description="Image path")

class BPAExportData(BaseModel):
    farm_id: str = Field(..., description="Farm ID")
    animal_data: BPAAnalysisData = Field(..., description="Animal analysis data")
    export_timestamp: datetime = Field(default_factory=datetime.now, description="Export timestamp")
    api_version: str = Field("1.0", description="API version")

# Validation and accuracy schemas
class AccuracyReport(BaseModel):
    total_validations: int = Field(..., description="Total number of validations")
    atc_score_accuracy: float = Field(..., description="ATC score accuracy")
    breed_classification_accuracy: float = Field(..., description="Breed classification accuracy")
    measurement_accuracy: float = Field(..., description="Measurement accuracy")
    average_confidence: float = Field(..., description="Average confidence")
    validation_period: Dict[str, datetime] = Field(..., description="Validation period")
    recommendations: List[str] = Field(..., description="Improvement recommendations")

class ValidationMetrics(BaseModel):
    mae: float = Field(..., description="Mean Absolute Error")
    mse: float = Field(..., description="Mean Squared Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    r2_score: float = Field(..., description="R-squared score")
    correlation: float = Field(..., description="Correlation coefficient")

# Configuration schemas
class AIConfig(BaseModel):
    detection_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    keypoint_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Keypoint detection confidence")
    breed_confidence: float = Field(0.6, ge=0.0, le=1.0, description="Breed classification confidence")
    disease_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Disease detection confidence")
    enable_aruco_detection: bool = Field(True, description="Enable ArUco marker detection")
    enable_measurements: bool = Field(True, description="Enable body measurements")
    enable_atc_scoring: bool = Field(True, description="Enable ATC scoring")

class SystemConfig(BaseModel):
    max_file_size: int = Field(10 * 1024 * 1024, description="Maximum file size in bytes")
    allowed_formats: List[str] = Field(["jpg", "jpeg", "png", "bmp"], description="Allowed image formats")
    processing_timeout: int = Field(300, description="Processing timeout in seconds")
    ai_config: AIConfig = Field(default_factory=AIConfig, description="AI configuration")

# Pagination schemas
class PaginationParams(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(10, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseModel):
    items: List[Any] = Field(..., description="Response items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")