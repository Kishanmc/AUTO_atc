"""
Database models for AutoATC system
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any

from .database import Base

class AnimalAnalysis(Base):
    """Main table for storing animal analysis records."""
    
    __tablename__ = "animal_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    animal_id = Column(String(100), unique=True, index=True, nullable=False)
    image_path = Column(String(500), nullable=False)
    original_filename = Column(String(255), nullable=False)
    
    # Analysis metadata
    analysis_date = Column(DateTime, default=func.now(), nullable=False)
    processing_time = Column(Float, nullable=True)  # in seconds
    
    # Detection results
    animal_detected = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True)
    bounding_box = Column(JSON, nullable=True)  # [x1, y1, x2, y2]
    
    # Keypoint detection
    keypoints_detected = Column(Boolean, default=False)
    keypoints_data = Column(JSON, nullable=True)  # List of keypoint coordinates
    
    # Measurements
    measurements = Column(JSON, nullable=True)  # Body measurements in cm
    
    # ATC Scoring
    atc_score = Column(Float, nullable=True)
    atc_grade = Column(String(10), nullable=True)  # A+, A, B+, B, etc.
    
    # Breed classification
    breed_predicted = Column(String(100), nullable=True)
    breed_confidence = Column(Float, nullable=True)
    
    # Disease detection
    diseases_detected = Column(JSON, nullable=True)  # List of detected diseases
    disease_confidence = Column(Float, nullable=True)
    
    # Status flags
    is_processed = Column(Boolean, default=False)
    is_exported_to_bpa = Column(Boolean, default=False)
    is_validated = Column(Boolean, default=False)
    
    # Validation data
    manual_atc_score = Column(Float, nullable=True)
    validation_notes = Column(Text, nullable=True)
    
    # Relationships
    results = relationship("AnalysisResult", back_populates="analysis")
    bpa_exports = relationship("BPAExport", back_populates="analysis")
    
    def __repr__(self):
        return f"<AnimalAnalysis(id={self.id}, animal_id='{self.animal_id}')>"

class AnalysisResult(Base):
    """Detailed analysis results for each animal."""
    
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("animal_analyses.id"), nullable=False)
    
    # Result type (detection, measurement, scoring, breed, disease)
    result_type = Column(String(50), nullable=False)
    
    # Raw results data
    raw_data = Column(JSON, nullable=False)
    
    # Processed results
    processed_data = Column(JSON, nullable=True)
    
    # Confidence scores
    confidence = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    analysis = relationship("AnimalAnalysis", back_populates="results")
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, type='{self.result_type}')>"

class BPAExport(Base):
    """BPA export records for tracking data synchronization."""
    
    __tablename__ = "bpa_exports"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("animal_analyses.id"), nullable=False)
    
    # Export metadata
    export_date = Column(DateTime, default=func.now(), nullable=False)
    bpa_animal_id = Column(String(100), nullable=True)  # ID assigned by BPA
    
    # Export status
    status = Column(String(20), default="pending")  # pending, success, failed
    error_message = Column(Text, nullable=True)
    
    # Export data
    exported_data = Column(JSON, nullable=False)
    
    # Retry information
    retry_count = Column(Integer, default=0)
    last_retry = Column(DateTime, nullable=True)
    
    # Relationships
    analysis = relationship("AnimalAnalysis", back_populates="bpa_exports")
    
    def __repr__(self):
        return f"<BPAExport(id={self.id}, status='{self.status}')>"

class ValidationRecord(Base):
    """Manual validation records for accuracy assessment."""
    
    __tablename__ = "validation_records"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("animal_analyses.id"), nullable=False)
    
    # Manual scores
    manual_atc_score = Column(Float, nullable=False)
    manual_breed = Column(String(100), nullable=True)
    manual_measurements = Column(JSON, nullable=True)
    
    # Validator information
    validator_id = Column(String(100), nullable=True)
    validator_notes = Column(Text, nullable=True)
    
    # Validation metadata
    validation_date = Column(DateTime, default=func.now())
    is_approved = Column(Boolean, default=False)
    
    # Accuracy metrics
    atc_score_difference = Column(Float, nullable=True)
    breed_match = Column(Boolean, nullable=True)
    measurement_accuracy = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<ValidationRecord(id={self.id}, analysis_id={self.analysis_id})>"
