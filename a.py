import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 示例：检查标注文件与图像是否匹配（以YOLO格式为例）
def plot_bboxes(image_path, label_path, class_names=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # 跳过空行或错误行
        class_id, x_center, y_center, width, height = map(float, parts[:5])

        # 转换为像素坐标
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if class_names:
            cv2.putText(image, class_names[int(class_id)], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()


# 调用示例
image_path = "/home/a10/slh/yolo/datasets/UAVDT/train/images/image00001.jpg"
label_path = "/home/a10/slh/yolo/datasets/UAVDT/train/images/labels/image00001.txt"
class_names = ["car", "truck", "bus"]  # 替换为你的类别名
plot_bboxes(image_path, label_path, class_names)



