import sys
sys.path.insert(0, "/home/houxiangguan/YOLOv8/other/ultralytics-main-no-problem/")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

import os

os.environ["WANDB_API_KEY"] = "5b002bccf3115458ea20c83fb51ff5f0b1d67e34"
os.environ["WANDB_MODE"] = "offline"


# Load a model  efficientnetlite0-mtl   yolov8-mobilenetv3
model = YOLO('/home/houxiangguan/YOLOv8/other/ultralytics-main-no-problem/ultralytics/cfg/models/v8/yolov8s.yaml', task='detect').load('yolov8s.pt')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights


# Train the model
if __name__ == '__main__':
    model.train(data='/home/houxiangguan/YOLOv8/other/ultralytics-main-no-problem/ultralytics/datasets/coco128.yaml', batch=16, epochs=100, imgsz=(416,416), device=[6], name='yolopm', val=True, task='detect')

