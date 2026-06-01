"""Train YOLOv8 on synthetic circuit diagram dataset."""

from ultralytics import YOLO
from pathlib import Path


def train():
    data_yaml = "E:/circuit_data/circuit_dataset.yaml"

    # Start from YOLOv8 nano pretrained weights
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=20,
        imgsz=320,
        batch=4,
        name="circuit_detector",
        exist_ok=True,
        device="cpu",
        workers=1,
        patience=5,
        verbose=True,
    )

    # Save trained model
    model.export(format="onnx")
    print(f"\nModel saved to: runs/detect/circuit_detector/weights/best.pt")
    return results


def test_on_image(model_path: str, image_path: str):
    """Run inference on a single image."""
    model = YOLO(model_path)
    results = model(image_path)
    results[0].show()
    results[0].save("data/images/detection_result.jpg")
    return results[0]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        model_path = "runs/detect/circuit_detector/weights/best.pt"
        img = sys.argv[2] if len(sys.argv) > 2 else "data/images/preview_led_good.png"
        test_on_image(model_path, img)
    else:
        train()
