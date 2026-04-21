# Bangladeshi Taka Note Detection

YOLOv8-based object detection REST API for identifying Bangladeshi currency denominations, containerized with Docker.

## Detected Classes

**Taka Notes (9):** 2_Taka, 5_Taka, 10_Taka, 20_Taka, 50_Taka, 100_Taka, 200_Taka, 500_Taka, 1000_Taka

**Coins (2):** 2_Taka_coin, 5_Taka_coin *(combined model only)*

## Project Structure

```
Bangladeshi_Taka_Detection_Project/
├── api/
│   ├── __init__.py
│   ├── app.py                # FastAPI application with /predict endpoint
│   └── inference.py          # Model loading and inference logic
├── models/
│   ├── taka_detection_best.pt        # YOLOv8 weights (notes only)
│   └── combined_detection_best.pt    # YOLOv8 weights (notes + coins)
├── test_images/              # Place test images here for API testing
├── inference_results/        # Annotated output images from inference
├── dataset/                  # Dataset source links
├── notebook/                 # Phase-1 training notebook
├── training_eval_logs/       # Training logs and evaluation metrics
├── inference_demo.py         # Standalone inference script (Task 1)
├── Dockerfile                # Docker container configuration
├── .dockerignore             # Files excluded from Docker build
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start (Local, without Docker)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the inference demo

Uses the combined model (notes + coins) by default:
```bash
python inference_demo.py --image test_images/sample1.jpg
```

To use the notes-only model instead:
```bash
python inference_demo.py --image test_images/sample1.jpg --model models/taka_detection_best.pt
```

### 3. Start the API server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at `http://localhost:8000`.

### 4. Test the API

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@test_images/sample1.jpg"
```

**Using Python requests:**
```python
import requests

url = "http://localhost:8000/predict"
with open("test_images/sample1.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})
print(response.json())
```

## API Reference

### `POST /predict`

Upload an image to detect Bangladeshi currency denominations.

**Request:** `multipart/form-data` with a `file` field (JPEG or PNG image)

**Response (200 OK):**
```json
{
  "success": true,
  "filename": "sample1.jpg",
  "detections_count": 3,
  "detections": [
    {
      "class_name": "100_Taka",
      "confidence": 0.9234,
      "bbox": { "x1": 120.5, "y1": 85.3, "x2": 450.2, "y2": 310.7 }
    }
  ]
}
```

**Error responses:**
- `400` – Invalid file type or empty file
- `500` – Inference failure

### `GET /health`

Returns `{"status": "healthy", "model_loaded": true}`.

### `GET /`

Returns a welcome message with usage instructions.

## Docker Deployment

### Build the Docker image

```bash
docker build -t taka-detection-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 --name taka-api taka-detection-api
```

The API is now accessible at `http://localhost:8000`.

### Test the containerized API

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@test_images/sample1.jpg"
```

### Stop and remove the container

```bash
docker stop taka-api
docker rm taka-api
```

## API Testing Guide

1. Place  **test images** of Bangladeshi currency notes in the `test_images/` folder.
2. Start the API (locally or via Docker).
3. Use **Postman** or **curl** to send each image to `POST /predict`.

**Postman steps:**
1. Open Postman → New Request → **POST** `http://localhost:8000/predict`
2. Go to **Body** → select **form-data**
3. Key: `file`, Type: **File**, Value: select your test image
4. Click **Send**

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv8n (nano) |
| Input size | 640 × 640 |
| Training epochs | 100 |
| Optimizer | Auto (SGD) |
| Confidence threshold | 0.25 |
