# ML API Documentation

## Overview
This API provides prediction services for a machine learning model trained on California housing data. The API is containerized using Docker and provides REST endpoints for health checks, model information, and predictions.

## Architecture
- **Framework**: Flask
- **Model**: Decision Tree Regressor (best performing model with RMSE: 0.709)
- **Container**: Docker with Python 3.9
- **Model Storage**: Pickle files for portability

## API Endpoints

### 1. Health Check
- **URL**: `GET /health`
- **Description**: Check if the API is running
- **Response**: 
  ```json
  {
    "status": "healthy"
  }
  ```

### 2. Model Information
- **URL**: `GET /info`
- **Description**: Get information about the loaded model
- **Response**:
  ```json
  {
    "feature_count": 8,
    "model_loaded": true,
    "model_name": "Decision Tree",
    "rmse": 0.7090862381364729
  }
  ```

### 3. Prediction
- **URL**: `POST /predict`
- **Description**: Get predictions from the model
- **Content-Type**: `application/json`

#### Input Formats

**Option 1: Features Object**
```json
{
  "features": [8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23]
}
```

**Option 2: Array Format**
```json
[8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23]
```

#### Response
```json
{
  "prediction": [1.375],
  "status": "success"
}
```

## Feature Order
The model expects 8 features in the following order:
1. MedInc (median income)
2. HouseAge (house age)
3. AveRooms (average rooms)
4. AveBedrms (average bedrooms)
5. Population (population)
6. AveOccup (average occupancy)
7. Latitude (latitude)
8. Longitude (longitude)

## Running the API

### Using Docker
1. Build the image:
   ```bash
   docker build -t my-api .
   ```

2. Run the container:
   ```bash
   docker run -d -p 5000:5000 --name my-api-container my-api
   ```

3. Test the API:
   ```bash
   curl -X GET http://localhost:5000/health
   ```

### Testing
Run the comprehensive test suite:
```bash
python3 test_api.py
```

## Files Structure
- `src/api.py` - Main Flask application
- `src/models/train_and_track.py` - Model training and saving
- `models/best_model.pkl` - Trained model
- `models/model_info.pkl` - Model metadata
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `test_api.py` - API test suite

## Error Handling
The API includes comprehensive error handling:
- Model loading failures
- Invalid input formats
- Prediction errors
- Detailed logging for debugging

## Performance
- Model: Decision Tree Regressor
- RMSE: 0.709
- Response time: < 100ms for single predictions
- Supports both single and batch predictions

## Security Notes
- This is a development server configuration
- For production, use a proper WSGI server like Gunicorn
- Consider adding authentication for production deployments