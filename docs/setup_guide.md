# AutoATC Setup Guide

## Overview

This guide will help you set up the AutoATC system for development and production environments. The system consists of a FastAPI backend, Streamlit frontend, AI modules, and database components.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for AI models)
- **Storage**: Minimum 20GB free space
- **GPU**: Optional but recommended for faster AI processing

### Software Dependencies

- Docker and Docker Compose
- Git
- Python 3.9+
- Node.js 16+ (for future React frontend)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/autoatc.git
cd autoatc
```

### 2. Environment Setup

Create environment file:

```bash
cp .env.example .env
```

Edit `.env` file with your configuration:

```env
# Database
DATABASE_URL=sqlite:///./atc.db
# For PostgreSQL: postgresql://user:password@localhost/atc_db

# API Configuration
DEBUG=True
HOST=0.0.0.0
PORT=8000

# BPA Integration
BPA_API_URL=https://api.bpa.gov.in/v1
BPA_API_KEY=your_bpa_api_key

# AI Models
DETECTION_MODEL_PATH=models/yolov8n.pt
BREED_MODEL_PATH=models/breed_classifier.pt
DISEASE_MODEL_PATH=models/disease_detector.pt

# Security
SECRET_KEY=your-secret-key-change-in-production
```

### 3. Install Dependencies

#### Option A: Using Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

#### Option B: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend/streamlit_app
pip install -r requirements.txt
```

### 4. Initialize Database

```bash
# Using Docker
docker-compose exec backend python -c "from db.database import create_tables; create_tables()"

# Or locally
cd backend
python -c "from db.database import create_tables; create_tables()"
```

### 5. Download AI Models

```bash
# Create models directory
mkdir -p models

# Download YOLO model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt

# Download breed classifier (if available)
# wget https://your-model-url/breed_classifier.pt -O models/breed_classifier.pt

# Download disease detector (if available)
# wget https://your-model-url/disease_detector.pt -O models/disease_detector.pt
```

### 6. Start Services

#### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Local Development

```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend/streamlit_app
streamlit run app.py --server.port 8501
```

### 7. Access the Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Development Setup

### 1. Project Structure

```
AutoATC/
├── backend/                 # FastAPI backend
│   ├── main.py             # Application entry point
│   ├── routers/            # API routes
│   ├── models/             # Pydantic schemas
│   ├── db/                 # Database models
│   ├── services/           # Business logic
│   ├── utils/              # Utility functions
│   └── requirements.txt    # Python dependencies
├── ai_modules/             # AI/ML modules
│   ├── detection/          # Object detection
│   ├── measurement/        # Measurement calculation
│   ├── scoring/            # ATC scoring
│   ├── breed/              # Breed classification
│   └── disease/            # Disease detection
├── frontend/               # Frontend applications
│   └── streamlit_app/      # Streamlit interface
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── tests/                  # Test files
├── docker-compose.yml      # Docker configuration
└── README.md
```

### 2. Database Setup

#### SQLite (Development)

```bash
# Database file will be created automatically
# Location: backend/atc.db
```

#### PostgreSQL (Production)

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb atc_db
sudo -u postgres createuser atc_user
sudo -u postgres psql -c "ALTER USER atc_user PASSWORD 'atc_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE atc_db TO atc_user;"

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://atc_user:atc_password@localhost/atc_db
```

### 3. AI Models Setup

#### YOLO Model

```bash
# Download pre-trained YOLOv8 model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt

# For better accuracy, use YOLOv8m or YOLOv8l
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -O models/yolov8m.pt
```

#### Custom Models

```bash
# Train your own models using the provided training scripts
cd scripts/training
python train_breed_classifier.py
python train_disease_detector.py
```

### 4. Testing

```bash
# Run backend tests
cd backend
pytest tests/

# Run frontend tests
cd frontend/streamlit_app
pytest tests/

# Run integration tests
cd tests
pytest integration/
```

### 5. Code Quality

```bash
# Format code
black backend/ frontend/ scripts/
isort backend/ frontend/ scripts/

# Lint code
flake8 backend/ frontend/ scripts/
mypy backend/

# Run all quality checks
make lint
make format
make test
```

## Production Deployment

### 1. Docker Production Setup

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with production configuration
docker-compose -f docker-compose.prod.yml up -d
```

### 2. Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

### 3. Environment Configuration

#### Production Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:password@db-host:5432/atc_db

# Security
SECRET_KEY=your-very-secure-secret-key
DEBUG=False

# BPA Integration
BPA_API_URL=https://api.bpa.gov.in/v1
BPA_API_KEY=your-production-bpa-api-key

# Monitoring
LOG_LEVEL=INFO
ENABLE_MONITORING=True
```

### 4. SSL/TLS Setup

```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 -nodes

# Update nginx configuration
# Edit nginx/nginx.conf to enable HTTPS
```

### 5. Monitoring and Logging

```bash
# Set up monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access monitoring dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

## Configuration

### 1. AI Model Configuration

Edit `backend/config/settings.py`:

```python
# Detection settings
DETECTION_CONFIDENCE = 0.5
KEYPOINT_CONFIDENCE = 0.5

# Classification settings
BREED_CONFIDENCE = 0.6
DISEASE_CONFIDENCE = 0.5

# Processing settings
MAX_PROCESSING_TIME = 300
ENABLE_BACKGROUND_TASKS = True
```

### 2. Database Configuration

```python
# Database settings
DATABASE_URL = "postgresql://user:password@localhost/atc_db"
DATABASE_ECHO = False

# Connection pool settings
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
```

### 3. API Configuration

```python
# API settings
HOST = "0.0.0.0"
PORT = 8000
DEBUG = False

# CORS settings
ALLOWED_ORIGINS = [
    "https://your-frontend-domain.com",
    "https://your-admin-domain.com"
]
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error

```bash
# Check database status
docker-compose exec postgres pg_isready

# Check connection string
echo $DATABASE_URL

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

#### 2. AI Model Loading Error

```bash
# Check model files
ls -la models/

# Download missing models
python scripts/download_models.py

# Check model compatibility
python -c "import torch; print(torch.__version__)"
```

#### 3. Memory Issues

```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### 4. Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :8501

# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Backend
  - "8502:8501"  # Frontend
```

### Logs and Debugging

#### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# With timestamps
docker-compose logs -f -t
```

#### Debug Mode

```bash
# Enable debug mode
export DEBUG=True

# Run with debug logging
docker-compose -f docker-compose.debug.yml up
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_animal_analyses_animal_id ON animal_analyses(animal_id);
CREATE INDEX idx_animal_analyses_analysis_date ON animal_analyses(analysis_date);
CREATE INDEX idx_validation_records_analysis_id ON validation_records(analysis_id);
```

### 2. Caching

```bash
# Enable Redis caching
docker-compose up -d redis

# Configure cache settings
CACHE_TTL = 3600  # 1 hour
REDIS_URL = redis://redis:6379
```

### 3. Image Processing

```python
# Optimize image processing
MAX_IMAGE_SIZE = (1024, 1024)
IMAGE_QUALITY = 85
ENABLE_IMAGE_COMPRESSION = True
```

## Security Considerations

### 1. API Security

```python
# Enable API key authentication
ENABLE_API_AUTH = True
API_KEYS = ["your-api-key-1", "your-api-key-2"]

# Rate limiting
RATE_LIMIT_PER_MINUTE = 60
```

### 2. Data Security

```python
# Encrypt sensitive data
ENABLE_DATA_ENCRYPTION = True
ENCRYPTION_KEY = "your-encryption-key"

# Secure file uploads
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
```

### 3. Network Security

```bash
# Configure firewall
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

## Backup and Recovery

### 1. Database Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U atc_user atc_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
docker-compose exec -T postgres psql -U atc_user atc_db < backup_20240115_103000.sql
```

### 2. Model Backup

```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz models/

# Restore models
tar -xzf models_backup_20240115_103000.tar.gz
```

### 3. Configuration Backup

```bash
# Backup configuration
cp .env .env.backup
cp docker-compose.yml docker-compose.yml.backup
```

## Support and Maintenance

### 1. Regular Maintenance

```bash
# Update dependencies
docker-compose pull
docker-compose up -d

# Clean up old images
docker system prune -a

# Update models
python scripts/update_models.py
```

### 2. Monitoring

```bash
# Check system health
curl http://localhost:8000/health

# Monitor resource usage
docker stats

# Check logs for errors
docker-compose logs --tail=100 | grep ERROR
```

### 3. Support Contacts

- **Technical Support**: support@autoatc.gov.in
- **Documentation**: https://docs.autoatc.gov.in
- **Issue Tracker**: https://github.com/your-org/autoatc/issues

## Next Steps

1. **Customize Models**: Train models on your specific data
2. **Integrate BPA**: Set up BPA API integration
3. **Add Users**: Implement user management system
4. **Scale Deployment**: Set up load balancing and clustering
5. **Monitor Performance**: Implement comprehensive monitoring
6. **Add Features**: Extend functionality based on requirements
