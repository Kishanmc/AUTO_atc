"""
Export router for BPA integration and data export
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging
import httpx
import json
from datetime import datetime

from db.database import get_db
from db.models import AnimalAnalysis, BPAExport
from models.schemas import (
    ExportBPARequest, ExportResponse, BPAAnalysisData, BPAExportData,
    ErrorResponse
)
from utils.schema_mapper import BPASchemaMapper
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize schema mapper
bpa_mapper = BPASchemaMapper()
settings = get_settings()

@router.post("/export/bpa", response_model=ExportResponse)
async def export_to_bpa(
    request: ExportBPARequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Export analysis results to BPA (Bharat Pashudhan App).
    
    Args:
        request: Export request data
        background_tasks: Background tasks handler
        db: Database session
        
    Returns:
        Export response
    """
    try:
        # Get analysis record
        analysis = db.query(AnimalAnalysis).filter(
            AnimalAnalysis.animal_id == request.animal_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Check if already exported
        existing_export = db.query(BPAExport).filter(
            BPAExport.analysis_id == analysis.id,
            BPAExport.status == "success"
        ).first()
        
        if existing_export:
            return ExportResponse(
                success=True,
                message="Analysis already exported to BPA",
                bpa_animal_id=existing_export.bpa_animal_id,
                exported_data=existing_export.exported_data
            )
        
        # Prepare BPA data
        bpa_data = _prepare_bpa_data(analysis, request)
        
        # Validate BPA data
        validation = bpa_mapper.validate_bpa_data(bpa_data)
        if not validation['valid']:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid BPA data: {', '.join(validation['errors'])}"
            )
        
        # Create export record
        export_record = BPAExport(
            analysis_id=analysis.id,
            status="pending",
            exported_data=bpa_data
        )
        db.add(export_record)
        db.commit()
        db.refresh(export_record)
        
        # Start background export task
        background_tasks.add_task(
            _export_to_bpa_background,
            export_record.id,
            bpa_data,
            request.bpa_api_key
        )
        
        return ExportResponse(
            success=True,
            message="Export initiated successfully",
            exported_data=bpa_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating BPA export: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/export/bpa/{animal_id}/status")
async def get_export_status(
    animal_id: str,
    db: Session = Depends(get_db)
):
    """
    Get BPA export status for an animal.
    
    Args:
        animal_id: Animal identifier
        db: Database session
        
    Returns:
        Export status
    """
    try:
        # Get analysis record
        analysis = db.query(AnimalAnalysis).filter(
            AnimalAnalysis.animal_id == animal_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Get export records
        exports = db.query(BPAExport).filter(
            BPAExport.analysis_id == analysis.id
        ).order_by(BPAExport.export_date.desc()).all()
        
        if not exports:
            return {
                "animal_id": animal_id,
                "exported": False,
                "status": "not_exported",
                "message": "No export attempts found"
            }
        
        latest_export = exports[0]
        
        return {
            "animal_id": animal_id,
            "exported": latest_export.status == "success",
            "status": latest_export.status,
            "bpa_animal_id": latest_export.bpa_animal_id,
            "export_date": latest_export.export_date,
            "retry_count": latest_export.retry_count,
            "error_message": latest_export.error_message,
            "total_exports": len(exports)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting export status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get export status: {str(e)}")

@router.post("/export/bpa/{animal_id}/retry")
async def retry_bpa_export(
    animal_id: str,
    background_tasks: BackgroundTasks,
    bpa_api_key: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Retry BPA export for an animal.
    
    Args:
        animal_id: Animal identifier
        background_tasks: Background tasks handler
        bpa_api_key: Optional BPA API key
        db: Database session
        
    Returns:
        Retry response
    """
    try:
        # Get analysis record
        analysis = db.query(AnimalAnalysis).filter(
            AnimalAnalysis.animal_id == animal_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Get latest export record
        latest_export = db.query(BPAExport).filter(
            BPAExport.analysis_id == analysis.id
        ).order_by(BPAExport.export_date.desc()).first()
        
        if not latest_export:
            raise HTTPException(status_code=404, detail="No export record found")
        
        # Check retry limit
        max_retries = 3
        if latest_export.retry_count >= max_retries:
            raise HTTPException(
                status_code=400, 
                detail=f"Maximum retry limit ({max_retries}) exceeded"
            )
        
        # Update retry count
        latest_export.retry_count += 1
        latest_export.last_retry = datetime.now()
        latest_export.status = "pending"
        latest_export.error_message = None
        db.commit()
        
        # Start background retry task
        background_tasks.add_task(
            _export_to_bpa_background,
            latest_export.id,
            latest_export.exported_data,
            bpa_api_key
        )
        
        return {
            "message": f"Retry initiated (attempt {latest_export.retry_count}/{max_retries})",
            "animal_id": animal_id,
            "retry_count": latest_export.retry_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying BPA export: {e}")
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")

@router.get("/export/bpa/schema")
async def get_bpa_schema():
    """
    Get BPA schema information.
    
    Returns:
        BPA schema information
    """
    try:
        schema_info = bpa_mapper.get_bpa_schema_info()
        return schema_info
        
    except Exception as e:
        logger.error(f"Error getting BPA schema: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")

@router.post("/export/validate")
async def validate_export_data(
    data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Validate data for BPA export.
    
    Args:
        data: Data to validate
        db: Database session
        
    Returns:
        Validation results
    """
    try:
        # Map data to BPA format
        bpa_data = bpa_mapper.map_analysis_to_bpa(data)
        
        # Validate BPA data
        validation = bpa_mapper.validate_bpa_data(bpa_data)
        
        return {
            "valid": validation['valid'],
            "warnings": validation['warnings'],
            "errors": validation['errors'],
            "bpa_data": bpa_data if validation['valid'] else None
        }
        
    except Exception as e:
        logger.error(f"Error validating export data: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

async def _export_to_bpa_background(
    export_id: int,
    bpa_data: Dict[str, Any],
    api_key: Optional[str] = None
):
    """
    Background task to export data to BPA.
    
    Args:
        export_id: Export record ID
        bpa_data: BPA formatted data
        api_key: BPA API key
    """
    try:
        from db.database import SessionLocal
        
        db = SessionLocal()
        export_record = db.query(BPAExport).filter(BPAExport.id == export_id).first()
        
        if not export_record:
            logger.error(f"Export record {export_id} not found")
            return
        
        # Get BPA API configuration
        bpa_api_url = settings.BPA_API_URL
        bpa_api_key = api_key or settings.BPA_API_KEY
        
        if not bpa_api_url or not bpa_api_key:
            export_record.status = "failed"
            export_record.error_message = "BPA API configuration not found"
            db.commit()
            return
        
        # Prepare request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bpa_api_key}",
            "X-API-Version": "1.0"
        }
        
        # Make API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{bpa_api_url}/animals/analysis",
                json=bpa_data,
                headers=headers
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Update export record
                export_record.status = "success"
                export_record.bpa_animal_id = response_data.get("animal_id")
                export_record.error_message = None
                
                # Update analysis record
                analysis = db.query(AnimalAnalysis).filter(
                    AnimalAnalysis.id == export_record.analysis_id
                ).first()
                if analysis:
                    analysis.is_exported_to_bpa = True
                
                logger.info(f"Successfully exported to BPA: {export_record.bpa_animal_id}")
                
            else:
                # Handle API error
                error_message = f"BPA API error: {response.status_code} - {response.text}"
                export_record.status = "failed"
                export_record.error_message = error_message
                
                logger.error(f"BPA export failed: {error_message}")
            
            db.commit()
            
    except Exception as e:
        logger.error(f"Error in background BPA export: {e}")
        
        try:
            export_record.status = "failed"
            export_record.error_message = str(e)
            db.commit()
        except:
            pass
    
    finally:
        db.close()

def _prepare_bpa_data(analysis: AnimalAnalysis, request: ExportBPARequest) -> Dict[str, Any]:
    """
    Prepare analysis data for BPA export.
    
    Args:
        analysis: Analysis record
        request: Export request
        
    Returns:
        BPA formatted data
    """
    try:
        # Base analysis data
        analysis_data = {
            'animal_id': analysis.animal_id,
            'analysis_date': analysis.analysis_date,
            'atc_score': analysis.atc_score or 0.0,
            'atc_grade': analysis.atc_grade or 'N/A',
            'breed': analysis.breed_predicted,
            'confidence': analysis.confidence_score or 0.0,
            'image_path': analysis.image_path
        }
        
        # Add measurements if requested
        if request.include_measurements and analysis.measurements:
            analysis_data['measurements'] = analysis.measurements
        
        # Add diseases if requested
        if request.include_diseases and analysis.diseases_detected:
            analysis_data['diseases'] = analysis.diseases_detected
        
        # Map to BPA format
        bpa_data = bpa_mapper.map_analysis_to_bpa(analysis_data)
        
        return bpa_data
        
    except Exception as e:
        logger.error(f"Error preparing BPA data: {e}")
        raise ValueError(f"Failed to prepare BPA data: {str(e)}")