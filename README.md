ML Ops Assignment – Model Training, Tracking, and API (with Docker)

## Overview
This project trains regression models on the California Housing dataset, tracks experiments with MLflow, selects and saves the best model, and exposes a prediction API (Flask) that is containerized with Docker.

## Project Structure
- `src/data/` – data loading and preprocessing
- `src/models/train_and_track.py` – trains 2 models (Linear Regression, Decision Tree), logs to MLflow, selects best, saves pickle
- `src/api.py` – Flask API for predictions
- `models/` – saved model artifacts (`best_model.pkl`, `model_info.pkl`)
- `mlruns/` – local MLflow tracking data
- `requirements.txt` – pinned dependencies for Docker build
- `Dockerfile` – container definition (Python 3.9)
- `test_api.py` – simple API test suite

## Local Setup
1. Create and activate a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Train Models + Track with MLflow
Runs two models, logs metrics/artifacts to MLflow, selects the best by RMSE, registers in MLflow, and saves pickles for the API.

```bash
python3 src/models/train_and_track.py
```

Artifacts produced:
- `models/best_model.pkl`
- `models/model_info.pkl`

Optional: launch MLflow UI locally
```bash
mlflow ui --port 6006
```
Then visit `http://localhost:6006` to browse runs, metrics, and models.

## Run the API Locally
```bash
python3 src/api.py
```

Endpoints:
- Health: `GET /health`
- Model info: `GET /info`
- Predict: `POST /predict`

Examples:
```bash
# Health
curl -X GET http://localhost:5000/health

# Info
curl -X GET http://localhost:5000/info

# Predict (object format)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23]}'

# Predict (array format)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '[8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23]'
```

## Docker

Build image (uses Python 3.9 slim and pinned requirements):
```bash
docker build -t my-api .
```

Run container on port 5000:
```bash
docker run -d -p 5000:5000 --name my-api-container my-api
```

Run container on port 5555:
```bash
docker run -d -p 5555:5000 --name my-api-5555 my-api
```

Test endpoints (adjust port as needed):
```bash
curl -X GET http://localhost:5000/health
curl -X GET http://localhost:5000/info
```

Common container management:
```bash
docker ps                      # list running containers
docker logs my-api-container   # view logs
docker stop my-api-container   # stop
docker rm my-api-container     # remove
```

## API Test Suite
With the container running or local API started:
```bash
python3 test_api.py
```

## Troubleshooting
- 404 on `http://localhost:PORT/` – the root path `/` is not defined. Use `/health`, `/info`, or `/predict`.
- "Model not available" – ensure you ran `python3 src/models/train_and_track.py` so `models/best_model.pkl` exists and is included in the Docker build context.
- Pickle/NumPy/Sklearn compatibility – Docker uses pinned versions (see `requirements.txt`). If you retrain models, rebuild the image to align dependencies.
- MLflow warnings about deprecated params – harmless; we log models with input examples and register only the best model.

## Data Management with DVC
This repo tracks raw and processed data via DVC (`src/data/raw.dvc`, `src/data/processed.dvc`). The actual data files are not stored in Git; pull them with DVC or regenerate locally.

### Option A: Pull data with DVC (recommended)
1. Install DVC (choose extra based on your remote; if unsure, start with base):
   ```bash
   pip install dvc
   # or for specific remotes, e.g. S3 / GCS
   # pip install "dvc[s3]"  # AWS S3
   # pip install "dvc[gs]"  # Google Cloud Storage
   ```
2. Inspect configured remotes (optional):
   ```bash
   dvc remote list
   ```
3. Pull data artifacts referenced by `.dvc` files:
   ```bash
   dvc pull
   ```
   This will materialize the raw and processed datasets referenced by `src/data/raw.dvc` and `src/data/processed.dvc`.

If you do not have access to the DVC remote, use Option B.

### Option B: Regenerate data locally
Run the data scripts to recreate datasets without relying on DVC remotes.
```bash
python3 src/data/load_data.py       # downloads/loads source data
python3 src/data/preprocess.py     # builds processed dataset
```

### Helpful DVC commands
```bash
dvc status            # see if working tree matches DVC metadata
dvc pull              # fetch and checkout data from remote
dvc push              # (if you have write access) push new data to remote
dvc repro             # reproduce pipeline if dvc.yaml is present
```

## Notes
- The best model (by RMSE) is currently Decision Tree Regressor and is saved under `models/` for API use.
- The training script also registers the best model in MLflow’s model registry.

