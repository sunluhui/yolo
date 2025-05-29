import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO  # 或其他高性能检测器

# 初始化轻量级预训练模型 (YOLOv8n)
model = YOLO('yolov8n.pt')
background_frames = []

for img_path in tqdm('/home/a10/slh/yolo/datasets/UAVDT'):
    img = cv2.imread(img_path)
    results = model(img, conf=0.25, verbose=False)  # 降低置信度避免漏检
    if len(results[0].boxes) == 0:  # 无检测目标
        background_frames.append(img_path)