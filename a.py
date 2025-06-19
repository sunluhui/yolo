import os
import numpy as np
from PIL import Image

# 配置路径
image_dir = "/home/a10/slh/DOTAv1.5/images"
label_dir = "/home/a10/slh/DOTAv1.5/labels"
output_dir = "/home/a10/slh/labels_yolo"
os.makedirs(output_dir, exist_ok=True)

# DOTA类别映射YOLO ID（按字母顺序排列）
class_names = [
    'plane', 'ship', 'storage tank', 'baseball diamond',
    'tennis court', 'basketball court', 'ground track field', 'harbor',
    'bridge', 'large vehicle', 'small vehicle',
    'helicopter', 'roundabout', 'soccer ball field', 'swimming pool', 'container crane'  # DOTA v1.5有16类
]
class_to_id = {name: idx for idx, name in enumerate(class_names)}

# 遍历所有标注文件
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    base_name = os.path.splitext(label_file)[0]
    img_path = os.path.join(image_dir, base_name + ".png")  # 根据实际图像扩展名调整

    # 获取图像尺寸
    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    except FileNotFoundError:
        print(f"跳过 {label_file}：找不到对应图像")
        continue

    yolo_lines = []

    with open(os.path.join(label_dir, label_file), 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 跳过注释行和空行
        if line.startswith("imagesource") or line.startswith("gsd") or not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 9:
            continue

        # 解析旋转框坐标和类别
        points = list(map(float, parts[:8]))
        class_name = parts[8]
        difficulty = int(parts[9]) if len(parts) > 9 else 0

        # 跳过困难样本（可选）
        # if difficulty == 1:
        #     continue

        # 转换为水平矩形框
        xs = points[0::2]  # 所有x坐标
        ys = points[1::2]  # 所有y坐标
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # 计算中心点和宽高（归一化）
        cx = (x_min + x_max) / 2.0 / img_w
        cy = (y_min + y_max) / 2.0 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        # 确保坐标在[0,1]范围内
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        # 获取类别ID
        if class_name not in class_to_id:
            continue
        class_id = class_to_id[class_name]

        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # 写入YOLO格式文件
    output_path = os.path.join(output_dir, base_name + ".txt")
    with open(output_path, 'w') as f:
        f.write("\n".join(yolo_lines))

print("转换完成！YOLO标签保存在:", output_dir)