"""
FastAPI REST API for Bangladeshi Taka Note Detection.

Endpoints
---------
POST /predict   - Upload an image, receive detected denominations with
                  confidence scores and bounding-box coordinates.
GET  /health    - Simple health-check endpoint.
GET  /          - Welcome message with usage instructions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from api.inference import predict, get_model

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

app = FastAPI(
    title="Bangladeshi Taka Detection API",
    description="YOLOv8-based REST API for detecting Bangladeshi currency denominations.",
    version="1.0.0",
)


@app.on_event("startup")
async def load_model_on_startup():
    """Pre-load the model so the first request is fast."""
    get_model()


@app.get("/")
async def root():
    return {
        "message": "Bangladeshi Taka Detection API",
        "usage": "POST an image to /predict to get detection results.",
        "endpoints": {
            "/predict": "POST – Upload image for detection",
            "/health": "GET  – Health check",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image file (JPEG/PNG), runs YOLOv8 detection, and returns
    detected denomination names, confidence scores, and bounding boxes.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Accepted types: JPEG, PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict(image_bytes, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "filename": file.filename,
            "detections_count": len(result["detections"]),
            "detections": result["detections"],
            "annotated_image_path": result["annotated_image_path"],
        },
    )
