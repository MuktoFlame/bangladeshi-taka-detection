"""
Standalone inference demo for the Bangladeshi Taka Detection YOLOv8 model.
Loads the trained weights and runs detection on a single image,
printing results and saving an annotated output image.

Usage:
    python inference_demo.py --image path/to/image.jpg
    python inference_demo.py --image path/to/image.jpg --model models/combined_detection_best.pt
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_MODEL = "models/combined_detection_best.pt"
CONFIDENCE_THRESHOLD = 0.25


def run_inference(image_path: str, model_path: str = DEFAULT_MODEL) -> list[dict]:
    """Run YOLOv8 detection on a single image and return structured results."""
    model = YOLO(model_path)
    results = model(image_path, conf=CONFIDENCE_THRESHOLD)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(float(c), 2) for c in box.xyxy[0].tolist()],
        })
    return detections


def visualize_and_save(image_path: str, model_path: str, output_path: str) -> None:
    """Run inference, draw bounding boxes, and save the annotated image."""
    model = YOLO(model_path)
    results = model(image_path, conf=CONFIDENCE_THRESHOLD)[0]

    annotated = results.plot()
    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Bangladeshi Taka Detection – Inference Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to YOLO .pt weights")
    parser.add_argument("--output", type=str, default=None, help="Path for annotated output image")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found at '{args.image}'")
        return

    if not Path(args.model).exists():
        print(f"Error: Model weights not found at '{args.model}'")
        return

    detections = run_inference(args.image, args.model)

    print(f"\n{'='*60}")
    print(f"  Inference Results for: {args.image}")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")

    if not detections:
        print("  No objects detected.")
    else:
        print(f"  {len(detections)} object(s) detected:\n")
        print(f"  {'Class':<18} {'Confidence':<12} {'BBox (x1,y1,x2,y2)'}")
        print(f"  {'-'*18} {'-'*12} {'-'*30}")
        for det in detections:
            bbox_str = ", ".join(f"{v:.1f}" for v in det["bbox"])
            print(f"  {det['class_name']:<18} {det['confidence']:<12.4f} [{bbox_str}]")

    print(f"{'='*60}\n")

    output_path = args.output or f"inference_results/demo_output_{Path(args.image).stem}.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    visualize_and_save(args.image, args.model, output_path)


if __name__ == "__main__":
    main()
