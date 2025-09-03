from ultralytics import YOLO

# Load a model
model = YOLO("/home/manu/tmp/yolo11n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx", imgsz=(384, 640), dynamic=True)
