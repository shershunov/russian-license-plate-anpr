import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")

model = YOLO("yolo26n.pt")

results = model.train(data="plates.yaml",
                      epochs=400,
                      imgsz=640,
                      patience=40,
                      batch=96,
                      hsv_v=0.5,
                      hsv_s=0.6,
                      mosaic=0.7,
                      fliplr=0.5,
                      degrees=5,
                      max_det=32,
                      pretrained=True,
                      plots=True,
                      device=[0, 1, 2, 3])
