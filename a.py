import os
import json
import cv2
import pandas as pd
from glob import glob


def convert_uavdt_to_coco(data_root, output_json):
    # 初始化COCO结构
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "vehicle"}]
    }

    # 遍历处理每个视频序列
    video_id = 1
    for seq in sorted(glob(os.path.join(data_root, 'M_attr/images/*'))):
        seq_name = os.path.basename(seq)
        image_dir = os.path.join(seq, 'img1')
        label_file = os.path.join(data_root, 'M_attr/GT', f"{seq_name}.txt")

        # 处理图像文件
        img_id = 1
        for img_path in sorted(glob(os.path.join(image_dir, '*.jpg'))):
            img_name = os.path.basename(img_path)
            img_info = {
                "id": img_id,
                "file_name": img_name,
                "width": cv2.imread(img_path).shape[1],
                "height": cv2.imread(img_path).shape[0]
            }
            coco["images"].append(img_info)

            # 处理标注文件
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8: continue

                    # 解析标注信息
                    x_center = float(parts[2]) * img_info["width"]
                    y_center = float(parts[3]) * img_info["height"]
                    width = float(parts[4]) * img_info["width"]
                    height = float(parts[5]) * img_info["height"]
                    category_id = 1  # UAVDT仅包含车辆类别

                    # 转换为COCO格式
                    annotation = {
                        "id": len(coco["annotations"]) + 1,
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [x_center - width / 2, y_center - height / 2, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    }
                    coco["annotations"].append(annotation)
            img_id += 1
        video_id += 1

    # 保存JSON文件
    with open(output_json, 'w') as f:
        json.dump(coco, f)


# 执行转换
convert_uavdt_to_coco('/path/to/M_attr', 'UAVDT_processed/annotations/instances_train.json')