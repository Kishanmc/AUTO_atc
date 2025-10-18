"""
AutoATC FastAPI Backend
Main application entry point for the Animal Type Classification system.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import os
import logging
from typing import Dict, Any

from routers import analyze, results, export
from db.database import init_db
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting AutoATC Backend...")
    await init_db()
    logger.info("Database initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AutoATC Backend...")

# Create FastAPI application
app = FastAPI(
    title="AutoATC API",
    description="AI-based Animal Type Classification system for cattle & buffaloes under Rashtriya Gokul Mission",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(results.router, prefix="/api/v1", tags=["Results"])
app.include_router(export.router, prefix="/api/v1", tags=["Export"])

# Mount static files for serving images and results
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AutoATC API - Animal Type Classification System",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AutoATC Backend",
        "version": "1.0.0"
    }

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint with detailed system information."""
    return {
        "api_status": "operational",
        "database": "connected",
        "ai_modules": {
            "detection": "ready",
            "measurement": "ready", 
            "scoring": "ready",
            "breed_classification": "ready",
            "disease_detection": "ready"
        },
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "results": "/api/v1/results/{animal_id}",
            "export_bpa": "/api/v1/export/bpa"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
