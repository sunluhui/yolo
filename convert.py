import os
from PIL import Image


def dota_to_yolo(dota_txt_path, yolo_txt_path, img_width, img_height):
    with open(dota_txt_path, 'r') as f_dota, open(yolo_txt_path, 'w') as f_yolo:
        for line in f_dota:
            parts = line.strip().split()
            # 提取坐标和类别
            coords = list(map(float, parts[:8]))
            class_id = int(parts[8])  # 假设类别已映射为整数

            # 解析四个顶点坐标
            x1, y1, x2, y2, x3, y3, x4, y4 = coords

            # 计算几何中心和宽高
            x_center = (x1 + x2 + x3 + x4) / 4
            y_center = (y1 + y2 + y3 + y4) / 4
            width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
            height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

            # 归一化
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            # 写入YOLO格式
            f_yolo.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 示例调用
            img_path = "images/train/0001.jpg"
            img = Image.open(img_path)
            img_width, img_height = img.size
            dota_to_yolo("labels/train/0001.txt", "labels/yolo/0001.txt", img_width, img_height)