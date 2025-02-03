# Reactor Foaming Classification System using Computer Vision (PyTorch)

This project implements a machine learning system for classifying reactor images as either "Foaming" or "Non-Foaming". The system uses a distributed architecture with a FastAPI backend for model serving and a Gradio frontend for user interaction.

## System Architecture

The system consists of two main components:

### 1. FastAPI Backend
- Serves a ResNet-18 model for binary classification
- Handles image preprocessing and inference
- Features:
  - AWS S3 integration for model storage
  - Image orientation correction
  - Health check endpoint
  - RESTful API for predictions
  - Containerized deployment

### 2. Gradio Frontend
- Provides an intuitive web interface
- Features:
  - Real-time image upload and prediction
  - Automatic communication with FastAPI backend
  - Logging system for debugging
  - Containerized deployment

## Project Structure

```
FoamingClassfication/
├── backend/
│   ├── model_backend.py    # FastAPI server implementation
│   ├── Dockerfile         # Backend container configuration
│   └── requirements.txt   # Backend dependencies
├── gradio-cdk/
│   ├── app.py            # Gradio frontend implementation
│   ├── Dockerfile        # Frontend container configuration
│   └── requirements.txt  # Frontend dependencies
```

## Prerequisites

- Docker
- AWS Account with:
  - S3 bucket access
  - AWS credentials
- Python 3.10

## Configuration

### Backend Environment Variables
Create a `.env` file in the backend directory:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
```

### Frontend Configuration
The API endpoint URL can be configured in `app.py`:
```python
API_URL = "http://your-backend-url:8000/predict/"
```

## Deployment

### Backend Deployment

1. Build the Docker image:
```bash
cd backend
docker build -t foaming-classifier-backend .
```

2. Run the container:
```bash
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  foaming-classifier-backend
```

### Frontend Deployment

1. Build the Docker image:
```bash
cd gradio-cdk
docker build -t foaming-classifier-frontend .
```

2. Run the container:
```bash
docker run -d -p 7860:7860 foaming-classifier-frontend
```

## API Endpoints

### Backend API
- `GET /health`
  - Health check endpoint
  - Returns: `{"status": "OK", "message": "FastAPI backend is running!"}`

- `POST /predict/`
  - Accepts: Multipart form data with image file
  - Returns: `{"prediction": "Foaming"/"Non-Foaming"}`

## Usage

1. Access the Gradio interface at `http://localhost:7860`
2. Upload a reactor image
3. The system will automatically process the image and display the classification result

## Model Details

- Architecture: ResNet-18
- Input Size: 224x224 pixels
- Classes: Binary (Foaming/Non-Foaming)
- Preprocessing:
  - Resize to 224x224
  - Normalize using ImageNet stats
  - EXIF orientation correction

## Error Handling

The system includes comprehensive error handling:
- Image validation
- API communication errors
- Model prediction errors
- Logging for debugging

## Monitoring

Both components include logging:
- Frontend: Gradio interface actions and API communication
- Backend: Model loading, predictions, and API requests

## Troubleshooting

1. Backend Issues:
   - Check the health endpoint: `curl http://localhost:8000/health`
   - Verify AWS credentials
   - Check Docker logs: `docker logs <backend-container-id>`

2. Frontend Issues:
   - Verify backend URL configuration
   - Check Docker logs: `docker logs <frontend-container-id>`
   - Verify port accessibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Your License Here]

## Authors

[Your Name/Organization]