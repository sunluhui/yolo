import os
import numpy as np


def remove_duplicate_labels(label_dir, iou_threshold=0.95):
    """删除重复标签，保留IOU最高的一个"""
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            path = os.path.join(label_dir, filename)
            with open(path, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:  # 单标签无需处理
                continue

            # 解析标签
            boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    box = list(map(float, parts[1:5]))
                    boxes.append((cls_id, box))

            # 计算IOU并标记重复项
            to_remove = set()
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if boxes[i][0] != boxes[j][0]:  # 不同类别跳过
                        continue

                    iou = calculate_iou(boxes[i][1], boxes[j][1])
                    if iou > iou_threshold:
                        # 保留置信度更高的（这里简单保留第一个）
                        to_remove.add(j)

            # 生成新标签
            new_lines = []
            for idx, (cls_id, box) in enumerate(boxes):
                if idx not in to_remove:
                    new_lines.append(f"{cls_id} {' '.join(map(str, box))}\n")

            # 保存清理后的标签
            with open(path, 'w') as f:
                f.writelines(new_lines)

            if to_remove:
                print(f"清理 {filename}: 移除 {len(to_remove)} 个重复标签")


def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""

    # 转换为中心坐标 -> 角坐标
    def xywh2xyxy(box):
        x, y, w, h = box
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    box1 = xywh2xyxy(box1)
    box2 = xywh2xyxy(box2)

    # 计算交集区域
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union


# 应用到训练集和验证集
train_label_dir = "/home/a10/slh/yolo/datasets/TinyPerson/labels/train"
val_label_dir = "/home/a10/slh/yolo/datasets/TinyPerson/labels/val"

print("正在清理训练集标签...")
remove_duplicate_labels(train_label_dir)

print("\n正在清理验证集标签...")
remove_duplicate_labels(val_label_dir)

print("\n重复标签清理完成！")