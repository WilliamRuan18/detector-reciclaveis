from ultralytics import YOLO

# modelo base
model = YOLO("yolov8n.pt")

# treinocxdfdfdfdfdgfgfgfgfgfggttybhgfgssdddkoefrr
model.train(
    data="dataset_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
