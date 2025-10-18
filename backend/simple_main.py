"""
Simplified AutoATC Backend - Minimal version for quick testing
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
import base64
from io import BytesIO
from PIL import Image
import uuid
from datetime import datetime

app = FastAPI(
    title="AutoATC API",
    description="AI-based Animal Type Classification system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for demo
analysis_results = {}

@app.get("/")
async def root():
    return {
        "message": "AutoATC API - Animal Type Classification System",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "AutoATC Backend",
        "version": "1.0.0"
    }

@app.get("/api/v1/status")
async def api_status():
    return {
        "api_status": "operational",
        "database": "in-memory",
        "ai_modules": {
            "detection": "ready",
            "measurement": "ready", 
            "scoring": "ready",
            "breed_classification": "ready",
            "disease_detection": "ready"
        }
    }

@app.post("/api/v1/analyze")
async def analyze_image(image_data: dict):
    """Analyze uploaded image and return mock results"""
    try:
        # Generate unique ID
        analysis_id = str(uuid.uuid4())
        
        # Mock analysis results
        mock_result = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "confidence": 0.85,
            "animal_type": "cattle",
            "processing_time": 2.5,
            "detection": {
                "bounding_box": {
                    "x": 100,
                    "y": 50,
                    "width": 300,
                    "height": 400,
                    "confidence": 0.92
                },
                "class": "cattle",
                "confidence": 0.92
            },
            "keypoints": {
                "head": {"x": 200, "y": 80, "confidence": 0.88},
                "neck": {"x": 200, "y": 150, "confidence": 0.85},
                "shoulder": {"x": 180, "y": 200, "confidence": 0.90},
                "back": {"x": 200, "y": 250, "confidence": 0.87},
                "hip": {"x": 200, "y": 350, "confidence": 0.89},
                "tail": {"x": 200, "y": 420, "confidence": 0.75}
            },
            "measurements": {
                "height": 120.5,
                "length": 180.2,
                "girth": 150.8,
                "width": 45.3,
                "scale_factor": 1.0,
                "confidence": 0.85
            },
            "atc_score": {
                "score": 78.5,
                "overall_score": 78.5,
                "grade": "A",
                "factors": {
                    "body_conformation": 80.0,
                    "muscle_development": 75.0,
                    "bone_structure": 82.0,
                    "overall_balance": 77.0,
                    "breed_characteristics": 80.0,
                    "health_indicators": 75.0
                },
                "recommendations": [
                    "Excellent body conformation",
                    "Good muscle development",
                    "Strong bone structure",
                    "Well-balanced proportions"
                ]
            },
            "breed_classification": {
                "breed": "Holstein Friesian",
                "predicted_breed": "Holstein Friesian",
                "confidence": 0.88,
                "alternative_breeds": [
                    {"breed": "Jersey", "confidence": 0.12},
                    {"breed": "Sahiwal", "confidence": 0.08}
                ]
            },
            "disease_detection": {
                "diseases_detected": [],
                "health_score": 85.0,
                "recommendations": [
                    "Animal appears healthy",
                    "Continue regular monitoring"
                ]
            },
            "annotated_image_url": f"/static/annotated_{analysis_id}.jpg"
        }
        
        # Store result
        analysis_results[analysis_id] = mock_result
        
        # Return in the format expected by frontend
        return {
            "status": "success",
            "data": mock_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/analyze/upload")
async def analyze_uploaded_file(file: UploadFile = File(...)):
    """Analyze uploaded file and return mock results"""
    try:
        # Read file content
        content = await file.read()
        
        # Generate unique ID
        analysis_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = f"static/upload_{analysis_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Mock analysis results (same as above)
        mock_result = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "confidence": 0.85,
            "animal_type": "cattle",
            "processing_time": 2.5,
            "detection": {
                "bounding_box": {
                    "x": 100,
                    "y": 50,
                    "width": 300,
                    "height": 400,
                    "confidence": 0.92
                },
                "class": "cattle",
                "confidence": 0.92
            },
            "keypoints": {
                "head": {"x": 200, "y": 80, "confidence": 0.88},
                "neck": {"x": 200, "y": 150, "confidence": 0.85},
                "shoulder": {"x": 180, "y": 200, "confidence": 0.90},
                "back": {"x": 200, "y": 250, "confidence": 0.87},
                "hip": {"x": 200, "y": 350, "confidence": 0.89},
                "tail": {"x": 200, "y": 420, "confidence": 0.75}
            },
            "measurements": {
                "height": 120.5,
                "length": 180.2,
                "girth": 150.8,
                "width": 45.3,
                "scale_factor": 1.0,
                "confidence": 0.85
            },
            "atc_score": {
                "score": 78.5,
                "overall_score": 78.5,
                "grade": "A",
                "factors": {
                    "body_conformation": 80.0,
                    "muscle_development": 75.0,
                    "bone_structure": 82.0,
                    "overall_balance": 77.0,
                    "breed_characteristics": 80.0,
                    "health_indicators": 75.0
                },
                "recommendations": [
                    "Excellent body conformation",
                    "Good muscle development",
                    "Strong bone structure",
                    "Well-balanced proportions"
                ]
            },
            "breed_classification": {
                "breed": "Holstein Friesian",
                "predicted_breed": "Holstein Friesian",
                "confidence": 0.88,
                "alternative_breeds": [
                    {"breed": "Jersey", "confidence": 0.12},
                    {"breed": "Sahiwal", "confidence": 0.08}
                ]
            },
            "disease_detection": {
                "diseases_detected": [],
                "health_score": 85.0,
                "recommendations": [
                    "Animal appears healthy",
                    "Continue regular monitoring"
                ]
            },
            "annotated_image_url": f"/static/annotated_{analysis_id}.jpg"
        }
        
        # Store result
        analysis_results[analysis_id] = mock_result
        
        # Return in the format expected by frontend
        return {
            "status": "success",
            "data": mock_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/v1/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Get analysis results by ID"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

@app.get("/api/v1/results")
async def list_results():
    """List all analysis results"""
    return {
        "results": list(analysis_results.values()),
        "total": len(analysis_results)
    }

if __name__ == "__main__":
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
