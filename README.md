# AutoATC - AI-based Animal Type Classification System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

AutoATC is a comprehensive AI-powered system for automatic classification and scoring of cattle and buffaloes under the Rashtriya Gokul Mission. The system provides end-to-end analysis including animal detection, breed classification, body measurements, ATC scoring, disease detection, and seamless integration with the Bharat Pashudhan App (BPA).

## 🚀 Features

### Core Functionality
- **Animal Detection**: YOLO-based detection of cattle and buffaloes in images
- **Breed Classification**: Deep learning models for identifying specific breeds
- **Body Measurements**: Precise calculation of physical measurements from keypoints
- **ATC Scoring**: Comprehensive Animal Type Classification scoring system
- **Disease Detection**: AI-powered health condition identification
- **BPA Integration**: Seamless export to Bharat Pashudhan App

### Technical Features
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Modern Frontend**: Streamlit interface with React.js dashboard (coming soon)
- **Database Support**: PostgreSQL for production, SQLite for development
- **Docker Support**: Complete containerization for easy deployment
- **Validation System**: Accuracy assessment and manual validation tools
- **Scalable Architecture**: Microservices-based design for horizontal scaling

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/autoatc.git
cd autoatc

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

cd ../frontend/streamlit_app
pip install -r requirements.txt

# Start services
# Terminal 1: Backend
cd backend
uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend/streamlit_app
streamlit run app.py
```

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Git
- 8GB+ RAM (16GB recommended for AI models)

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Memory**: Minimum 8GB RAM
- **Storage**: 20GB+ free space
- **GPU**: Optional but recommended for faster processing

### Step-by-Step Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/autoatc.git
   cd autoatc
   ```

2. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Download AI Models**
   ```bash
   mkdir -p models
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
   ```

4. **Start Services**
   ```bash
   docker-compose up --build
   ```

For detailed installation instructions, see [Setup Guide](docs/setup_guide.md).

## 💻 Usage

### Web Interface

1. **Upload Image**: Navigate to the Analysis page and upload an animal image
2. **Configure Analysis**: Select analysis options (breed classification, disease detection, measurements)
3. **View Results**: Review comprehensive analysis results including ATC score, breed, and measurements
4. **Export to BPA**: Export results directly to the Bharat Pashudhan App

### API Usage

```python
import requests
import base64

# Encode image
with open("animal_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Analyze image
response = requests.post("http://localhost:8000/api/v1/analyze", json={
    "image_data": image_data,
    "filename": "animal_image.jpg",
    "include_breed_classification": True,
    "include_disease_detection": True,
    "include_measurements": True
})

result = response.json()
print(f"ATC Score: {result['data']['atc_score']['score']}")
print(f"Breed: {result['data']['breed_classification']['breed']}")
```

### Command Line Tools

```bash
# Run validation analysis
python scripts/validation.py --days 30 --plots

# Import validation data
python scripts/data_import.py --action import --file validation_data.csv --type validation

# Generate sample data
python scripts/data_import.py --action create-sample --output sample_data.csv
```

## 📚 API Documentation

### Core Endpoints

- `POST /api/v1/analyze` - Analyze animal image
- `GET /api/v1/results/{animal_id}` - Get analysis results
- `POST /api/v1/export/bpa` - Export to BPA
- `GET /api/v1/results/accuracy-report` - Get accuracy report

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

For complete API documentation, see [API Specification](docs/api_spec.md).

## 🏗️ Architecture

### System Components

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
```

### AI Pipeline

1. **Image Preprocessing**: Resize, enhance, and normalize images
2. **Animal Detection**: YOLO-based object detection
3. **Keypoint Detection**: MediaPipe-based anatomical landmark detection
4. **Breed Classification**: Deep learning model for breed identification
5. **Measurement Calculation**: Precise body measurements from keypoints
6. **ATC Scoring**: Comprehensive scoring algorithm
7. **Disease Detection**: Health condition identification
8. **Result Integration**: Combine all analysis results

For detailed architecture information, see [Architecture Documentation](docs/architecture.md).

## ⚙️ Configuration

### Environment Variables

```env
# Database
DATABASE_URL=sqlite:///./atc.db

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
```

### AI Model Configuration

```python
# Detection settings
DETECTION_CONFIDENCE = 0.5
KEYPOINT_CONFIDENCE = 0.5

# Classification settings
BREED_CONFIDENCE = 0.6
DISEASE_CONFIDENCE = 0.5
```

## 🚀 Deployment

### Docker Deployment

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# With monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

### Cloud Deployment

The system is designed to be cloud-native and can be deployed on:
- AWS (ECS, EKS)
- Google Cloud (GKE, Cloud Run)
- Azure (AKS, Container Instances)
- DigitalOcean (Kubernetes)

## 🧪 Testing

### Run Tests

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend/streamlit_app
pytest tests/

# Integration tests
cd tests
pytest integration/
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=backend tests/
pytest --cov=frontend tests/
```

## 📊 Validation and Accuracy

### Validation System

The system includes comprehensive validation tools:

- **Manual Validation**: Compare AI results with expert assessments
- **Accuracy Metrics**: Calculate precision, recall, and correlation
- **Performance Reports**: Generate detailed accuracy reports
- **Continuous Improvement**: Identify areas for model enhancement

### Running Validation

```bash
# Generate accuracy report
python scripts/validation.py --days 30 --plots

# Import validation data
python scripts/data_import.py --action import --file validation_data.csv --type validation
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Quality

```bash
# Format code
black backend/ frontend/ scripts/
isort backend/ frontend/ scripts/

# Lint code
flake8 backend/ frontend/ scripts/
mypy backend/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Rashtriya Gokul Mission** for supporting this initiative
- **Ultralytics** for YOLO models
- **MediaPipe** for pose estimation
- **FastAPI** and **Streamlit** communities
- **Open source contributors** who made this project possible

## 📞 Support

- **Documentation**: [docs.autoatc.gov.in](https://docs.autoatc.gov.in)
- **Issues**: [GitHub Issues](https://github.com/your-org/autoatc/issues)
- **Email**: support@autoatc.gov.in
- **Discord**: [AutoATC Community](https://discord.gg/autoatc)

## 🗺️ Roadmap

### Version 1.1 (Q2 2024)
- [ ] Mobile app integration
- [ ] Real-time video analysis
- [ ] Advanced analytics dashboard
- [ ] Multi-language support

### Version 1.2 (Q3 2024)
- [ ] 3D pose estimation
- [ ] Multi-animal detection
- [ ] IoT device integration
- [ ] Blockchain integration

### Version 2.0 (Q4 2024)
- [ ] Advanced AI models
- [ ] Predictive analytics
- [ ] Global deployment
- [ ] Enterprise features

---

**AutoATC** - Empowering livestock management through AI technology 🐄🤖
