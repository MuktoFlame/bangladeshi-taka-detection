"""
Inference engine — loads the YOLO model once and exposes a reusable
predict function for the FastAPI application.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "combined_detection_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "inference_results"
CONFIDENCE_THRESHOLD = 0.25

_model: YOLO | None = None


def get_model() -> YOLO:
    """Lazy-load the YOLO model (singleton)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def predict(image_bytes: bytes, filename: str | None = None) -> dict:
    """
    Run YOLOv8 detection on raw image bytes.

    Returns a dict with:
      - detections: list of {class_name, confidence, bbox}
      - annotated_image_path: relative path to the saved annotated image
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Make sure the file is a valid JPEG or PNG.")

    model = get_model()
    results = model(img_bgr, conf=CONFIDENCE_THRESHOLD)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": {
                "x1": round(float(box.xyxy[0][0]), 2),
                "y1": round(float(box.xyxy[0][1]), 2),
                "x2": round(float(box.xyxy[0][2]), 2),
                "y2": round(float(box.xyxy[0][3]), 2),
            },
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem if filename else "upload"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"api_{stem}_{timestamp}.jpg"

    annotated_bgr = results.plot()
    cv2.imwrite(str(output_path), annotated_bgr)

    return {
        "detections": detections,
        "annotated_image_path": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
