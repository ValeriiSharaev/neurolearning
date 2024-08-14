from ultralytics import YOLO

model = YOLO("yolov8n-obb.pt")

results = model.train(data="signs-obb3/dataset.yaml", imgsz=640, epochs=100)
