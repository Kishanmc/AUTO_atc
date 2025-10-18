"""
Results router for retrieving and managing analysis results
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from datetime import datetime, timedelta

from db.database import get_db
from db.models import AnimalAnalysis, AnalysisResult, ValidationRecord
from models.schemas import (
    ResultsResponse, PaginatedResponse, PaginationParams,
    ValidationResponse, AccuracyReport, ErrorResponse
)
from utils.schema_mapper import ValidationSchemaMapper

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize schema mapper
validation_mapper = ValidationSchemaMapper()

@router.get("/results/{animal_id}", response_model=ResultsResponse)
async def get_analysis_results(
    animal_id: str,
    db: Session = Depends(get_db)
):
    """
    Get analysis results for a specific animal.
    
    Args:
        animal_id: Animal identifier
        db: Database session
        
    Returns:
        Analysis results
    """
    try:
        # Get analysis record
        analysis = db.query(AnimalAnalysis).filter(
            AnimalAnalysis.animal_id == animal_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Get detailed results
        results = db.query(AnalysisResult).filter(
            AnalysisResult.analysis_id == analysis.id
        ).all()
        
        # Convert to response format
        analysis_result = _convert_analysis_to_result(analysis, results)
        
        return ResultsResponse(
            animal_id=animal_id,
            analysis=analysis_result,
            created_at=analysis.analysis_date,
            updated_at=analysis.analysis_date
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")

@router.get("/results", response_model=PaginatedResponse)
async def list_analysis_results(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    sort_by: Optional[str] = Query(None, description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    animal_type: Optional[str] = Query(None, description="Filter by animal type"),
    breed: Optional[str] = Query(None, description="Filter by breed"),
    min_atc_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum ATC score"),
    max_atc_score: Optional[float] = Query(None, ge=0, le=100, description="Maximum ATC score"),
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    db: Session = Depends(get_db)
):
    """
    List analysis results with filtering and pagination.
    
    Args:
        page: Page number
        size: Page size
        sort_by: Sort field
        sort_order: Sort order
        animal_type: Filter by animal type
        breed: Filter by breed
        min_atc_score: Minimum ATC score
        max_atc_score: Maximum ATC score
        date_from: Filter from date
        date_to: Filter to date
        db: Database session
        
    Returns:
        Paginated list of analysis results
    """
    try:
        # Build query
        query = db.query(AnimalAnalysis)
        
        # Apply filters
        if animal_type:
            query = query.filter(AnimalAnalysis.animal_type == animal_type)
        
        if breed:
            query = query.filter(AnimalAnalysis.breed_predicted == breed)
        
        if min_atc_score is not None:
            query = query.filter(AnimalAnalysis.atc_score >= min_atc_score)
        
        if max_atc_score is not None:
            query = query.filter(AnimalAnalysis.atc_score <= max_atc_score)
        
        if date_from:
            query = query.filter(AnimalAnalysis.analysis_date >= date_from)
        
        if date_to:
            query = query.filter(AnimalAnalysis.analysis_date <= date_to)
        
        # Apply sorting
        if sort_by:
            if hasattr(AnimalAnalysis, sort_by):
                sort_column = getattr(AnimalAnalysis, sort_by)
                if sort_order == "desc":
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())
            else:
                # Default sorting
                query = query.order_by(AnimalAnalysis.analysis_date.desc())
        else:
            query = query.order_by(AnimalAnalysis.analysis_date.desc())
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * size
        analyses = query.offset(offset).limit(size).all()
        
        # Convert to response format
        items = []
        for analysis in analyses:
            results = db.query(AnalysisResult).filter(
                AnalysisResult.analysis_id == analysis.id
            ).all()
            
            analysis_result = _convert_analysis_to_result(analysis, results)
            items.append(analysis_result)
        
        # Calculate pagination info
        pages = (total + size - 1) // size
        
        return PaginatedResponse(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages
        )
        
    except Exception as e:
        logger.error(f"Error listing analysis results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")

@router.get("/results/{animal_id}/validation", response_model=ValidationResponse)
async def get_validation_results(
    animal_id: str,
    db: Session = Depends(get_db)
):
    """
    Get validation results for a specific animal.
    
    Args:
        animal_id: Animal identifier
        db: Database session
        
    Returns:
        Validation results
    """
    try:
        # Get analysis record
        analysis = db.query(AnimalAnalysis).filter(
            AnimalAnalysis.animal_id == animal_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Get validation record
        validation = db.query(ValidationRecord).filter(
            ValidationRecord.analysis_id == analysis.id
        ).first()
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        return ValidationResponse(
            animal_id=animal_id,
            validation_id=validation.id,
            accuracy_metrics={
                'atc_score_accuracy': validation.atc_score_difference or 0.0,
                'breed_classification_accuracy': 1.0 if validation.breed_match else 0.0,
                'measurement_accuracy': validation.measurement_accuracy or 0.0
            },
            atc_score_difference=validation.atc_score_difference or 0.0,
            breed_match=validation.breed_match,
            measurement_accuracy=validation.measurement_accuracy,
            validation_date=validation.validation_date
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting validation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve validation: {str(e)}")

@router.get("/results/accuracy-report", response_model=AccuracyReport)
async def get_accuracy_report(
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    db: Session = Depends(get_db)
):
    """
    Get accuracy report for the specified period.
    
    Args:
        days: Number of days to include in report
        db: Database session
        
    Returns:
        Accuracy report
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get validation records
        validations = db.query(ValidationRecord).filter(
            ValidationRecord.validation_date >= start_date,
            ValidationRecord.validation_date <= end_date
        ).all()
        
        if not validations:
            return AccuracyReport(
                total_validations=0,
                atc_score_accuracy=0.0,
                breed_classification_accuracy=0.0,
                measurement_accuracy=0.0,
                average_confidence=0.0,
                validation_period={'start': start_date.isoformat(), 'end': end_date.isoformat()},
                recommendations=['No validation data available for the specified period']
            )
        
        # Convert to validation data format
        validation_data = []
        for validation in validations:
            validation_data.append({
                'atc_score_accuracy': 1.0 - (validation.atc_score_difference or 0.0) / 100.0,
                'breed_classification_accuracy': 1.0 if validation.breed_match else 0.0,
                'measurement_accuracy': validation.measurement_accuracy or 0.0,
                'confidence': 0.8,  # Default confidence
                'created_at': validation.validation_date.isoformat()
            })
        
        # Generate accuracy report
        report_data = validation_mapper.generate_accuracy_report(validation_data)
        
        return AccuracyReport(
            total_validations=report_data['total_validations'],
            atc_score_accuracy=report_data['atc_score_accuracy'],
            breed_classification_accuracy=report_data['breed_classification_accuracy'],
            measurement_accuracy=report_data['measurement_accuracy'],
            average_confidence=report_data['average_confidence'],
            validation_period=report_data['validation_period'],
            recommendations=report_data['recommendations']
        )
        
    except Exception as e:
        logger.error(f"Error generating accuracy report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.delete("/results/{animal_id}")
async def delete_analysis_results(
    animal_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete analysis results for a specific animal.
    
    Args:
        animal_id: Animal identifier
        db: Database session
        
    Returns:
        Deletion confirmation
    """
    try:
        # Get analysis record
        analysis = db.query(AnimalAnalysis).filter(
            AnimalAnalysis.animal_id == animal_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Delete related records
        db.query(AnalysisResult).filter(
            AnalysisResult.analysis_id == analysis.id
        ).delete()
        
        db.query(ValidationRecord).filter(
            ValidationRecord.analysis_id == analysis.id
        ).delete()
        
        # Delete analysis record
        db.delete(analysis)
        db.commit()
        
        return {"message": f"Analysis results for animal {animal_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis results: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete results: {str(e)}")

def _convert_analysis_to_result(analysis: AnimalAnalysis, results: List[AnalysisResult]) -> dict:
    """Convert database analysis to result format."""
    try:
        # Group results by type
        results_by_type = {}
        for result in results:
            results_by_type[result.result_type] = result
        
        # Build analysis result
        analysis_result = {
            'animal_id': analysis.animal_id,
            'animal_type': analysis.breed_predicted or 'unknown',
            'confidence': analysis.confidence_score or 0.0,
            'bounding_box': analysis.bounding_box,
            'keypoints': analysis.keypoints_data or [],
            'measurements': analysis.measurements,
            'atc_score': {
                'score': analysis.atc_score or 0.0,
                'grade': analysis.atc_grade or 'N/A',
                'factors': {},
                'recommendations': []
            } if analysis.atc_score else None,
            'breed_classification': {
                'breed': analysis.breed_predicted,
                'confidence': analysis.breed_confidence or 0.0,
                'alternative_breeds': []
            } if analysis.breed_predicted else None,
            'diseases': analysis.diseases_detected or [],
            'processing_time': analysis.processing_time or 0.0,
            'analysis_date': analysis.analysis_date,
            'status': 'completed' if analysis.is_processed else 'processing'
        }
        
        # Add detailed results if available
        if 'atc_scoring' in results_by_type:
            atc_data = results_by_type['atc_scoring'].raw_data
            analysis_result['atc_score'] = atc_data
        
        if 'breed_classification' in results_by_type:
            breed_data = results_by_type['breed_classification'].raw_data
            analysis_result['breed_classification'] = breed_data
        
        if 'disease_detection' in results_by_type:
            disease_data = results_by_type['disease_detection'].raw_data
            analysis_result['diseases'] = disease_data.get('diseases', [])
        
        return analysis_result
        
    except Exception as e:
        logger.warning(f"Error converting analysis to result: {e}")
        return {
            'animal_id': analysis.animal_id,
            'animal_type': 'unknown',
            'confidence': 0.0,
            'bounding_box': None,
            'keypoints': [],
            'measurements': None,
            'atc_score': None,
            'breed_classification': None,
            'diseases': [],
            'processing_time': 0.0,
            'analysis_date': analysis.analysis_date,
            'status': 'error'
        }