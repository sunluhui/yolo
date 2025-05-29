import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 示例：检查标注文件与图像是否匹配（以YOLO格式为例）
def check_data_consistency(data_dir, label_dir):
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    # 检查图像和标注文件是否一一对应
    for img_file in image_files:
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        if label_file not in label_files:
            print(f"Missing label for image: {img_file}")

    print(f"Total images: {len(image_files)}, Total labels: {len(label_files)}")


# 调用示例
data_dir = "/home/a10/slh/yolo/datasets/UAVDT/train/images"
label_dir = "/home/a10/slh/yolo/datasets/UAVDT/train/labels"
check_data_consistency(data_dir, label_dir)