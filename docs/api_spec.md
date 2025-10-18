# AutoATC API Specification

## Overview

The AutoATC API provides comprehensive endpoints for animal image analysis, breed classification, ATC scoring, and data export. The API follows RESTful principles and uses JSON for data exchange.

## Base URL

```
Development: http://localhost:8000/api/v1
Production: https://api.autoatc.gov.in/v1
```

## Authentication

Currently, the API uses API key authentication for BPA integration. Future versions will include JWT-based authentication for user management.

### Headers

```http
Content-Type: application/json
Authorization: Bearer <api_key>  # For BPA integration
```

## Endpoints

### 1. Health Check

#### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "AutoATC Backend",
  "version": "1.0.0"
}
```

### 2. Analysis Endpoints

#### POST /analyze

Analyze an animal image for ATC scoring, breed classification, and disease detection.

**Request Body:**
```json
{
  "image_data": "base64_encoded_image_string",
  "filename": "animal_image.jpg",
  "animal_id": "ATC001",  // Optional
  "include_breed_classification": true,
  "include_disease_detection": true,
  "include_measurements": true
}
```

**Response:**
```json
{
  "data": {
    "animal_id": "ATC001",
    "animal_type": "cattle",
    "confidence": 0.85,
    "bounding_box": {
      "x1": 100.0,
      "y1": 150.0,
      "x2": 400.0,
      "y2": 500.0
    },
    "keypoints": [
      {
        "id": 0,
        "name": "nose",
        "x": 250.0,
        "y": 200.0,
        "z": 0.0,
        "visibility": 0.9,
        "confidence": 0.85
      }
    ],
    "measurements": {
      "height": 120.5,
      "length": 150.2,
      "width": 60.8,
      "girth": 180.3,
      "leg_length": 75.0,
      "head_length": 45.2,
      "tail_length": 35.8,
      "bmi_estimate": 2.1,
      "body_condition_score": 3.5,
      "symmetry_score": 0.85
    },
    "atc_score": {
      "score": 85.2,
      "grade": "A",
      "factors": {
        "body_conformation": 82.5,
        "muscle_development": 88.0,
        "bone_structure": 80.0,
        "overall_balance": 85.5,
        "breed_characteristics": 90.0,
        "health_indicators": 85.0
      },
      "recommendations": [
        "Excellent ATC score - maintain current care practices",
        "Consider breeding program for genetic improvement"
      ]
    },
    "breed_classification": {
      "breed": "gir",
      "confidence": 0.92,
      "alternative_breeds": [
        {"breed": "sahival", "confidence": 0.15},
        {"breed": "haryana", "confidence": 0.08}
      ]
    },
    "diseases": [
      {
        "name": "Dermatitis",
        "category": "skin_conditions",
        "confidence": 0.75,
        "severity": "medium",
        "symptoms": ["inflammation", "redness"],
        "treatment": "Anti-inflammatory treatment",
        "description": "Skin inflammation detected"
      }
    ],
    "processing_time": 2.34,
    "analysis_date": "2024-01-15T10:30:00Z",
    "status": "completed"
  },
  "annotated_image_url": "/static/annotated/ATC001_annotated_20240115_103000.jpg",
  "message": "Analysis completed successfully for animal ATC001"
}
```

#### POST /analyze/upload

Alternative endpoint for file upload instead of base64 data.

**Request:** Multipart form data
- `file`: Image file
- `animal_id`: Optional animal ID
- `include_breed_classification`: Boolean
- `include_disease_detection`: Boolean
- `include_measurements`: Boolean

**Response:** Same as `/analyze`

#### GET /analyze/status/{animal_id}

Get the status of an ongoing analysis.

**Response:**
```json
{
  "animal_id": "ATC001",
  "status": "completed",
  "processing_time": 2.34,
  "analysis_date": "2024-01-15T10:30:00Z",
  "is_processed": true
}
```

### 3. Results Endpoints

#### GET /results/{animal_id}

Get analysis results for a specific animal.

**Response:**
```json
{
  "animal_id": "ATC001",
  "analysis": {
    // Same structure as analysis response
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

#### GET /results

List analysis results with filtering and pagination.

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Page size (default: 10, max: 100)
- `sort_by`: Sort field
- `sort_order`: Sort order (asc/desc)
- `animal_type`: Filter by animal type
- `breed`: Filter by breed
- `min_atc_score`: Minimum ATC score
- `max_atc_score`: Maximum ATC score
- `date_from`: Filter from date
- `date_to`: Filter to date

**Response:**
```json
{
  "items": [
    {
      "animal_id": "ATC001",
      "analysis": {
        // Analysis data
      }
    }
  ],
  "total": 150,
  "page": 1,
  "size": 10,
  "pages": 15
}
```

#### DELETE /results/{animal_id}

Delete analysis results for a specific animal.

**Response:**
```json
{
  "message": "Analysis results for animal ATC001 deleted successfully"
}
```

### 4. Export Endpoints

#### POST /export/bpa

Export analysis results to BPA (Bharat Pashudhan App).

**Request Body:**
```json
{
  "animal_id": "ATC001",
  "bpa_api_key": "your_bpa_api_key",  // Optional
  "include_measurements": true,
  "include_diseases": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Export initiated successfully",
  "bpa_animal_id": "BPA_12345",
  "exported_data": {
    "animalId": "ATC001",
    "analysisDate": "2024-01-15T10:30:00Z",
    "atcScore": 85.2,
    "atcGrade": "A",
    "breed": "gir",
    "confidenceScore": 0.85,
    "measurements": {
      "heightCm": 120.5,
      "bodyLengthCm": 150.2,
      "bodyWidthCm": 60.8,
      "chestGirthCm": 180.3,
      "units": "cm"
    },
    "healthStatus": [
      {
        "diseaseName": "Dermatitis",
        "diseaseCategory": "skin_conditions",
        "detectionConfidence": 0.75,
        "severityLevel": "medium",
        "symptoms": ["inflammation", "redness"],
        "recommendedTreatment": "Anti-inflammatory treatment"
      }
    ]
  }
}
```

#### GET /export/bpa/{animal_id}/status

Get BPA export status for an animal.

**Response:**
```json
{
  "animal_id": "ATC001",
  "exported": true,
  "status": "success",
  "bpa_animal_id": "BPA_12345",
  "export_date": "2024-01-15T10:35:00Z",
  "retry_count": 0,
  "error_message": null,
  "total_exports": 1
}
```

#### POST /export/bpa/{animal_id}/retry

Retry BPA export for an animal.

**Query Parameters:**
- `bpa_api_key`: Optional BPA API key

**Response:**
```json
{
  "message": "Retry initiated (attempt 1/3)",
  "animal_id": "ATC001",
  "retry_count": 1
}
```

### 5. Validation Endpoints

#### GET /results/{animal_id}/validation

Get validation results for a specific animal.

**Response:**
```json
{
  "animal_id": "ATC001",
  "validation_id": 123,
  "accuracy_metrics": {
    "atc_score_accuracy": 0.92,
    "breed_classification_accuracy": 1.0,
    "measurement_accuracy": 0.88
  },
  "atc_score_difference": 2.5,
  "breed_match": true,
  "measurement_accuracy": 0.88,
  "validation_date": "2024-01-15T11:00:00Z"
}
```

#### GET /results/accuracy-report

Get accuracy report for the specified period.

**Query Parameters:**
- `days`: Number of days to include (default: 30)

**Response:**
```json
{
  "total_validations": 150,
  "atc_score_accuracy": 0.89,
  "breed_classification_accuracy": 0.92,
  "measurement_accuracy": 0.85,
  "average_confidence": 0.87,
  "validation_period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  },
  "recommendations": [
    "ATC scoring accuracy is good - maintain current model",
    "Consider collecting more breed training data",
    "Measurement accuracy could be improved with better keypoint detection"
  ]
}
```

### 6. Configuration Endpoints

#### GET /export/bpa/schema

Get BPA schema information.

**Response:**
```json
{
  "field_mappings": {
    "animal_id": "animalId",
    "analysis_date": "analysisDate",
    "atc_score": "atcScore",
    "atc_grade": "atcGrade",
    "breed": "breed",
    "measurements": "measurements",
    "diseases": "healthStatus",
    "confidence": "confidenceScore",
    "image_path": "imagePath"
  },
  "measurement_mappings": {
    "height": "heightCm",
    "length": "bodyLengthCm",
    "width": "bodyWidthCm",
    "girth": "chestGirthCm",
    "leg_length": "legLengthCm",
    "head_length": "headLengthCm",
    "tail_length": "tailLengthCm"
  },
  "disease_mappings": {
    "name": "diseaseName",
    "category": "diseaseCategory",
    "confidence": "detectionConfidence",
    "severity": "severityLevel",
    "symptoms": "symptoms",
    "treatment": "recommendedTreatment",
    "description": "diseaseDescription"
  },
  "supported_fields": [
    "animal_id", "analysis_date", "atc_score", "atc_grade",
    "breed", "measurements", "diseases", "confidence", "image_path"
  ],
  "version": "1.0"
}
```

## Error Responses

### Standard Error Format

```json
{
  "error": "Error message",
  "detail": "Detailed error description",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Access denied
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

### Common Error Examples

#### 400 Bad Request
```json
{
  "error": "Invalid image data",
  "detail": "Image data must be valid base64 encoded string",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### 404 Not Found
```json
{
  "error": "Analysis not found",
  "detail": "No analysis found for animal ID: ATC999",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### 422 Unprocessable Entity
```json
{
  "error": "Validation error",
  "detail": {
    "image_data": ["Field required"],
    "filename": ["String too short"]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Analysis endpoints**: 10 requests per second per IP
- **Results endpoints**: 30 requests per second per IP
- **Export endpoints**: 5 requests per second per IP

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1642248600
```

## Pagination

List endpoints support pagination with the following parameters:

- `page`: Page number (1-based)
- `size`: Number of items per page (1-100)
- `sort_by`: Field to sort by
- `sort_order`: Sort direction (asc/desc)

Pagination metadata is included in responses:
```json
{
  "items": [...],
  "total": 150,
  "page": 1,
  "size": 10,
  "pages": 15
}
```

## Data Types

### Animal Types
- `cattle` - Cattle
- `buffalo` - Buffalo
- `unknown` - Unknown/undetermined

### ATC Grades
- `A+` - Excellent (90-100)
- `A` - Very Good (80-89)
- `B+` - Good (70-79)
- `B` - Fair (60-69)
- `C` - Poor (50-59)
- `D` - Very Poor (0-49)

### Severity Levels
- `low` - Low severity
- `medium` - Medium severity
- `high` - High severity
- `critical` - Critical severity

### Analysis Status
- `pending` - Analysis pending
- `processing` - Analysis in progress
- `completed` - Analysis completed
- `failed` - Analysis failed

## Webhooks

Future versions will support webhooks for:
- Analysis completion notifications
- Export status updates
- Validation results
- System alerts

## SDKs and Libraries

Official SDKs are planned for:
- Python
- JavaScript/TypeScript
- Java
- C#

## Changelog

### Version 1.0.0 (2024-01-15)
- Initial API release
- Basic analysis endpoints
- BPA integration
- Validation system
- Documentation
