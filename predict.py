import sys
sys.path.insert(0, "G:/论文/论文/yolov8论文/代码/ultralytics-main-no-problem")

from ultralytics import YOLO

model = YOLO('G:/论文/论文/yolov8论文/代码/ultralytics-main-no-problem/ultralytics/runs/detect/YOLOv8-soft-diou-nms-small/weights/best.pt')
model.predict(source='input',
              imgsz=(416, 416), device=[0], name='test_data', save=True, conf=0.65, iou=0.7,
              show_labels=True)
