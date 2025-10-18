# AutoATC System Architecture

## Overview

AutoATC (Automatic Animal Type Classification) is an AI-powered system for classifying and scoring cattle and buffaloes under the Rashtriya Gokul Mission. The system provides comprehensive analysis including animal detection, breed classification, body measurements, ATC scoring, disease detection, and BPA integration.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Modules    │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │   Database      │    │   Models        │
│   (Reverse      │    │   (PostgreSQL)  │    │   (YOLO, etc.)  │
│    Proxy)       │    └─────────────────┘    └─────────────────┘
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   External      │
│   (BPA API)     │
└─────────────────┘
```

## Components

### 1. Frontend Layer

#### Streamlit Application
- **Purpose**: User interface for image upload and analysis
- **Technology**: Streamlit, Python
- **Features**:
  - Image upload and preview
  - Analysis configuration
  - Results visualization
  - Export to BPA
  - Validation interface

#### Future Web Dashboard
- **Purpose**: Advanced web-based interface
- **Technology**: React.js, TypeScript
- **Features**:
  - Real-time analysis
  - Advanced filtering
  - Batch processing
  - Analytics dashboard

### 2. Backend Layer

#### FastAPI Application
- **Purpose**: REST API server and business logic
- **Technology**: FastAPI, Python
- **Features**:
  - Image processing endpoints
  - Analysis orchestration
  - Data persistence
  - BPA integration
  - Validation management

#### API Endpoints
- `POST /api/v1/analyze` - Analyze animal image
- `GET /api/v1/results/{animal_id}` - Get analysis results
- `GET /api/v1/results` - List analysis results
- `POST /api/v1/export/bpa` - Export to BPA
- `GET /api/v1/validation` - Get validation data

### 3. AI Modules Layer

#### Detection Module
- **Animal Detector**: YOLO-based detection of cattle/buffalo
- **Keypoint Detector**: MediaPipe-based anatomical keypoint detection
- **ArUco Detector**: Scale reference marker detection

#### Classification Module
- **Breed Classifier**: Deep learning model for breed classification
- **Disease Detector**: Health condition detection

#### Analysis Module
- **Measurement Calculator**: Body measurement calculation from keypoints
- **ATC Scorer**: Animal Type Classification scoring algorithm

### 4. Data Layer

#### Database
- **Primary**: PostgreSQL for production
- **Development**: SQLite for local development
- **Tables**:
  - `animal_analyses` - Main analysis records
  - `analysis_results` - Detailed results
  - `validation_records` - Manual validation data
  - `bpa_exports` - BPA export tracking

#### Storage
- **Images**: Local file system or cloud storage
- **Models**: Local model files or model registry
- **Cache**: Redis for session and result caching

### 5. Integration Layer

#### BPA Integration
- **Purpose**: Export data to Bharat Pashudhan App
- **Technology**: REST API, JSON schema mapping
- **Features**:
  - Automatic data export
  - Schema validation
  - Retry mechanism
  - Status tracking

#### External Services
- **Model Serving**: Optional model serving infrastructure
- **Monitoring**: System health and performance monitoring
- **Logging**: Centralized logging and error tracking

## Data Flow

### 1. Image Analysis Flow

```
User Upload → Frontend → Backend API → AI Pipeline → Database → Response
     │              │           │           │           │
     │              │           │           │           ▼
     │              │           │           │    ┌─────────────┐
     │              │           │           │    │   Results   │
     │              │           │           │    │   Storage   │
     │              │           │           │    └─────────────┘
     │              │           │           │
     │              │           │           ▼
     │              │           │    ┌─────────────┐
     │              │           │    │   AI        │
     │              │           │    │   Modules   │
     │              │           │    └─────────────┘
     │              │           │
     │              │           ▼
     │              │    ┌─────────────┐
     │              │    │   Image     │
     │              │    │   Processing│
     │              │    └─────────────┘
     │              │
     ▼              ▼
┌─────────────┐ ┌─────────────┐
│   Image     │ │   Analysis  │
│   Upload    │ │   Results   │
└─────────────┘ └─────────────┘
```

### 2. Validation Flow

```
Manual Validation → Validation Script → Accuracy Analysis → Report Generation
        │                  │                  │                  │
        │                  │                  │                  ▼
        │                  │                  │           ┌─────────────┐
        │                  │                  │           │   Accuracy  │
        │                  │                  │           │   Report    │
        │                  │                  │           └─────────────┘
        │                  │                  │
        │                  │                  ▼
        │                  │           ┌─────────────┐
        │                  │           │   Metrics   │
        │                  │           │   Calculation│
        │                  │           └─────────────┘
        │                  │
        │                  ▼
        │           ┌─────────────┐
        │           │   Data      │
        │           │   Import    │
        │           └─────────────┘
        │
        ▼
┌─────────────┐
│   Manual    │
│   Scores    │
└─────────────┘
```

## Security Architecture

### 1. Authentication & Authorization
- API key-based authentication for BPA integration
- Role-based access control for different user types
- JWT tokens for session management

### 2. Data Security
- Image data encryption at rest
- Secure API communication (HTTPS)
- Input validation and sanitization
- SQL injection prevention

### 3. Infrastructure Security
- Container security best practices
- Network segmentation
- Regular security updates
- Vulnerability scanning

## Scalability Considerations

### 1. Horizontal Scaling
- Stateless backend services
- Load balancer for multiple instances
- Database read replicas
- Caching layer for performance

### 2. Performance Optimization
- Image compression and resizing
- Model optimization and quantization
- Batch processing capabilities
- Asynchronous processing

### 3. Monitoring & Observability
- Application performance monitoring
- Log aggregation and analysis
- Health checks and alerting
- Metrics collection and visualization

## Deployment Architecture

### 1. Development Environment
- Local Docker containers
- SQLite database
- Mock external services

### 2. Production Environment
- Kubernetes cluster
- PostgreSQL database
- Redis cache
- Nginx reverse proxy
- SSL/TLS termination

### 3. CI/CD Pipeline
- Automated testing
- Container image building
- Deployment automation
- Rollback capabilities

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.9+
- **Database**: PostgreSQL, SQLite
- **Cache**: Redis
- **ORM**: SQLAlchemy

### Frontend
- **Framework**: Streamlit, React.js
- **Language**: Python, TypeScript
- **UI Components**: Custom components

### AI/ML
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: PyTorch, YOLO
- **Image Processing**: PIL, scikit-image
- **Data Science**: NumPy, Pandas

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes
- **Reverse Proxy**: Nginx
- **Monitoring**: Prometheus, Grafana

## Future Enhancements

### 1. Advanced AI Features
- Multi-animal detection
- 3D pose estimation
- Real-time video analysis
- Mobile app integration

### 2. Data Analytics
- Predictive analytics
- Trend analysis
- Performance dashboards
- Machine learning insights

### 3. Integration Expansion
- Additional government systems
- Third-party APIs
- IoT device integration
- Blockchain integration

### 4. User Experience
- Mobile-responsive design
- Offline capabilities
- Multi-language support
- Accessibility features
