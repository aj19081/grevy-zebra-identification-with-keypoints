from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/ultralytics/cfg/models/v8/yolov8-pose.yaml")
model = YOLO("yolov8s-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="data/grevy-zebra.yaml", epochs=100, imgsz=640)
