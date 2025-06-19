import os
import cv2
import numpy as np


def obb_to_yolo(obb_points, img_w, img_h):
    """
    将旋转框四点坐标转换为YOLO格式的水平矩形框
    Args:
        obb_points: [x1,y1, x2,y2, x3,y3, x4,y4]
        img_w: 图像宽度
        img_h: 图像高度
    Returns:
        [x_center, y_center, width, height] (归一化)
    """
    points = np.array(obb_points, dtype=np.float32).reshape(4, 2)

    # 计算水平外接矩形
    x_min = min(points[:, 0])
    x_max = max(points[:, 0])
    y_min = min(points[:, 1])
    y_max = max(points[:, 1])

    # 计算中心点和宽高（归一化）
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return [x_center, y_center, width, height]


def convert_obb_to_yolo(obb_label_dir, image_dir, output_dir, class_list):
    """
    批量转换OBB标注为YOLO格式
    Args:
        obb_label_dir: OBB标注文件目录（每个图像对应一个.txt文件）
        image_dir: 图像文件目录
        output_dir: 输出YOLO标注目录
        class_list: 类别名称列表
    """
    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(obb_label_dir):
        if not label_file.endswith(".txt"):
            continue

        # 读取图像尺寸
        img_name = label_file.replace(".txt", ".png")  # 根据实际扩展名调整
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found, skipping...")
            continue

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # 解析OBB标注
        with open(os.path.join(obb_label_dir, label_file), "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        yolo_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) < 9:
                continue

            # 提取旋转框坐标和类别
            obb = list(map(float, parts[:8]))
            class_name = parts[8]
            difficult = int(parts[9]) if len(parts) > 9 else 0

            # 跳过困难样本（可选）
            if difficult == 1:
                continue

            # 获取类别ID
            if class_name not in class_list:
                print(f"Warning: Unknown class {class_name}, skipping...")
                continue
            class_id = class_list.index(class_name)

            # 转换为YOLO格式
            yolo_box = obb_to_yolo(obb, img_w, img_h)
            yolo_lines.append(f"{class_id} {' '.join(map(str, yolo_box))}\n")

        # 保存YOLO标注文件
        output_path = os.path.join(output_dir, label_file)
        with open(output_path, "w") as f:
            f.writelines(yolo_lines)


if __name__ == "__main__":
    # 配置路径和类别
    OBB_LABEL_DIR = "/home/a10/slh/yolo/DOTAv1.5/labels"  # OBB标注目录
    IMAGE_DIR = "/home/a10/slh/yolo/DOTAv1.5/images"  # 图像目录
    OUTPUT_DIR = "/home/a10/slh/yolo/DOTAv1.5/yolo_labels"  # 输出目录

    # 加载类别列表
    with open("classes.txt", "r") as f:
        CLASSES = [line.strip() for line in f.readlines()]

    # 执行转换
    convert_obb_to_yolo(OBB_LABEL_DIR, IMAGE_DIR, OUTPUT_DIR, CLASSES)