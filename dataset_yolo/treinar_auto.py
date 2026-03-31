from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
        data="dataset_yolo/dataset.yaml",
        epochs=30,
        imgsz=640,
        batch=8
    )       