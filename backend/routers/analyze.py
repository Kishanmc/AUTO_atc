"""
Analysis router for animal image processing and classification
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
import base64
import io
import logging
from datetime import datetime
import time

from db.database import get_db
from db.models import AnimalAnalysis, AnalysisResult
from models.schemas import (
    AnalyzeRequest, AnalyzeResponse, AnalysisResult as AnalysisResultSchema,
    AnalysisStatus, AnimalType, BoundingBox, KeypointData, MeasurementData,
    ATCScore, BreedClassification, DiseaseDetection, ErrorResponse
)
from services.atc_pipeline import ATCPipeline
from services.breed_classifier import BreedClassifier
from services.disease_detector import DiseaseDetector
from utils.image_utils import process_image, save_image, generate_annotated_image

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize AI services
atc_pipeline = ATCPipeline()
breed_classifier = BreedClassifier()
disease_detector = DiseaseDetector()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_animal(
    request: AnalyzeRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze an animal image for ATC scoring, breed classification, and disease detection.
    
    This endpoint processes uploaded animal images and returns comprehensive analysis
    including body measurements, ATC scoring, breed classification, and disease detection.
    """
    start_time = time.time()
    
    try:
        # Decode base64 image data
        try:
            image_data = base64.b64decode(request.image_data)
            image = io.BytesIO(image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Generate animal ID if not provided
        animal_id = request.animal_id or f"animal_{int(datetime.now().timestamp())}"
        
        # Process and save image
        processed_image = process_image(image)
        image_path = save_image(processed_image, animal_id, request.filename)
        
        # Create analysis record
        analysis_record = AnimalAnalysis(
            animal_id=animal_id,
            image_path=image_path,
            original_filename=request.filename,
            analysis_date=datetime.now()
        )
        db.add(analysis_record)
        db.commit()
        db.refresh(analysis_record)
        
        # Run AI pipeline
        logger.info(f"Starting analysis for animal {animal_id}")
        
        # 1. Animal Detection
        detection_result = atc_pipeline.detect_animal(processed_image)
        analysis_record.animal_detected = detection_result['detected']
        analysis_record.confidence_score = detection_result['confidence']
        
        if detection_result['detected']:
            analysis_record.bounding_box = detection_result['bounding_box']
            
            # 2. Keypoint Detection
            keypoint_result = atc_pipeline.detect_keypoints(processed_image, detection_result['bounding_box'])
            analysis_record.keypoints_detected = keypoint_result['detected']
            analysis_record.keypoints_data = keypoint_result['keypoints']
            
            # 3. Body Measurements
            if request.include_measurements and keypoint_result['detected']:
                measurement_result = atc_pipeline.calculate_measurements(
                    processed_image, 
                    keypoint_result['keypoints']
                )
                analysis_record.measurements = measurement_result['measurements']
                
                # Store measurement results
                measurement_record = AnalysisResult(
                    analysis_id=analysis_record.id,
                    result_type="measurement",
                    raw_data=measurement_result,
                    processed_data=measurement_result['measurements'],
                    confidence=measurement_result['confidence']
                )
                db.add(measurement_record)
            
            # 4. ATC Scoring
            atc_result = atc_pipeline.calculate_atc_score(
                processed_image,
                keypoint_result['keypoints'],
                measurement_result.get('measurements', {}) if request.include_measurements else {}
            )
            analysis_record.atc_score = atc_result['score']
            analysis_record.atc_grade = atc_result['grade']
            
            # Store ATC results
            atc_record = AnalysisResult(
                analysis_id=analysis_record.id,
                result_type="atc_scoring",
                raw_data=atc_result,
                processed_data=atc_result,
                confidence=atc_result['confidence']
            )
            db.add(atc_record)
            
            # 5. Breed Classification
            if request.include_breed_classification:
                breed_result = breed_classifier.classify_breed(processed_image)
                analysis_record.breed_predicted = breed_result['breed']
                analysis_record.breed_confidence = breed_result['confidence']
                
                # Store breed results
                breed_record = AnalysisResult(
                    analysis_id=analysis_record.id,
                    result_type="breed_classification",
                    raw_data=breed_result,
                    processed_data=breed_result,
                    confidence=breed_result['confidence']
                )
                db.add(breed_record)
            
            # 6. Disease Detection
            if request.include_disease_detection:
                disease_result = disease_detector.detect_diseases(processed_image)
                analysis_record.diseases_detected = disease_result['diseases']
                analysis_record.disease_confidence = disease_result['confidence']
                
                # Store disease results
                disease_record = AnalysisResult(
                    analysis_id=analysis_record.id,
                    result_type="disease_detection",
                    raw_data=disease_result,
                    processed_data=disease_result,
                    confidence=disease_result['confidence']
                )
                db.add(disease_record)
        
        # Update processing status
        processing_time = time.time() - start_time
        analysis_record.processing_time = processing_time
        analysis_record.is_processed = True
        
        db.commit()
        db.refresh(analysis_record)
        
        # Generate annotated image
        annotated_image_url = None
        if detection_result['detected']:
            annotated_image_url = generate_annotated_image(
                processed_image, 
                detection_result, 
                keypoint_result if keypoint_result['detected'] else None,
                animal_id
            )
        
        # Prepare response data
        response_data = AnalysisResultSchema(
            animal_id=animal_id,
            animal_type=AnimalType.CATTLE if analysis_record.breed_predicted else AnimalType.UNKNOWN,
            confidence=analysis_record.confidence_score or 0.0,
            bounding_box=BoundingBox(**detection_result['bounding_box']) if detection_result['detected'] else None,
            keypoints=[
                KeypointData(**kp) for kp in keypoint_result['keypoints']
            ] if keypoint_result['detected'] else [],
            measurements=MeasurementData(**analysis_record.measurements) if analysis_record.measurements else None,
            atc_score=ATCScore(
                score=analysis_record.atc_score or 0.0,
                grade=analysis_record.atc_grade or "N/A",
                factors=atc_result.get('factors', {}),
                recommendations=atc_result.get('recommendations', [])
            ) if analysis_record.atc_score else None,
            breed_classification=BreedClassification(
                breed=analysis_record.breed_predicted,
                confidence=analysis_record.breed_confidence or 0.0,
                alternative_breeds=breed_result.get('alternative_breeds', [])
            ) if analysis_record.breed_predicted else None,
            diseases=[
                DiseaseDetection(**disease) for disease in analysis_record.diseases_detected or []
            ],
            processing_time=processing_time,
            analysis_date=analysis_record.analysis_date,
            status=AnalysisStatus.COMPLETED
        )
        
        logger.info(f"Analysis completed for animal {animal_id} in {processing_time:.2f}s")
        
        return AnalyzeResponse(
            data=response_data,
            annotated_image_url=annotated_image_url,
            message=f"Analysis completed successfully for animal {animal_id}"
        )
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        
        # Update analysis record with error status
        if 'analysis_record' in locals():
            analysis_record.is_processed = False
            db.commit()
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/analyze/upload", response_model=AnalyzeResponse)
async def analyze_animal_upload(
    file: UploadFile = File(...),
    animal_id: Optional[str] = Form(None),
    include_breed_classification: bool = Form(True),
    include_disease_detection: bool = Form(True),
    include_measurements: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Analyze an animal image uploaded as a file.
    
    Alternative endpoint for file uploads instead of base64 data.
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Encode to base64
        image_data = base64.b64encode(file_content).decode('utf-8')
        
        # Create request object
        request = AnalyzeRequest(
            image_data=image_data,
            filename=file.filename,
            animal_id=animal_id,
            include_breed_classification=include_breed_classification,
            include_disease_detection=include_disease_detection,
            include_measurements=include_measurements
        )
        
        # Process using main analyze endpoint
        return await analyze_animal(request, db)
        
    except Exception as e:
        logger.error(f"Error in file upload analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"File upload analysis failed: {str(e)}"
        )

@router.get("/analyze/status/{animal_id}")
async def get_analysis_status(
    animal_id: str,
    db: Session = Depends(get_db)
):
    """Get the status of an ongoing analysis."""
    analysis = db.query(AnimalAnalysis).filter(AnimalAnalysis.animal_id == animal_id).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {
        "animal_id": animal_id,
        "status": "completed" if analysis.is_processed else "processing",
        "processing_time": analysis.processing_time,
        "analysis_date": analysis.analysis_date,
        "is_processed": analysis.is_processed
    }
