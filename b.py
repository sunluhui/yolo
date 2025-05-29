import os
import json
import cv2
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# 配置参数
image_dir = "/home/a10/slh/yolo/datasets/OpenDataLab___UAVDT/raw/UAV-benchmark-M"
gt_dir = "/home/a10/slh/yolo/datasets/OpenDataLab___UAVDT/raw/UAV-benchmark-MOTD_v1.0/GT"
output_json = "/home/a10/slh/yolo/datasets/uavdt_coco.json"
img_size = (1920, 1080)  # 根据实际调整

# 类别映射（根据实际标注文件调整）
class_mapping = {"car": 0, "truck": 1, "bus": 2}

# 初始化COCO结构
coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": k, "name": v} for v, k in class_mapping.items()]
}

# 图片信息缓存
img_id_map = {}
ann_id = 1

# 处理图片文件
print("整理图片文件...")
os.makedirs("processed_images", exist_ok=True)
for seq in os.listdir(image_dir):
    seq_path = os.path.join(image_dir, seq)
    if not os.path.isdir(seq_path):
        continue
    for img_file in os.listdir(seq_path):
        src_path = os.path.join(seq_path, img_file)
        dst_path = os.path.join("processed_images", f"{seq}_{img_file}")

        # 重命名并移动图片
        cv2.imwrite(dst_path, cv2.imread(src_path))

        # 记录图片信息
        img_id = len(coco["images"]) + 1
        img_info = {
            "id": img_id,
            "file_name": dst_path,
            "width": img_size[0],
            "height": img_size[1]
        }
        coco["images"].append(img_info)
        img_id_map[f"{seq}_{img_file}"] = img_id

# 处理标注文件
print("转换标注文件...")
for seq in os.listdir(gt_dir):
    gt_path = os.path.join(gt_dir, seq)
    if not os.path.isfile(gt_path):
        continue

    # 读取标注数据
    df = pd.read_csv(gt_path, sep=' ', header=None,
                     names=['frame', 'tid', 'x', 'y', 'w', 'h', 'occl', 'vis', 'cls'])

    # 过滤无效数据
    df = df[df['cls'].isin(class_mapping.keys())]

    # 转换为绝对坐标
    for _, row in df.iterrows():
        img_key = f"{seq}_{row['frame']:06d}.jpg"
        if img_key not in img_id_map:
            continue

        img_id = img_id_map[img_key]
        x_center = row['x'] * img_size[0]
        y_center = row['y'] * img_size[1]
        width = row['w'] * img_size[0]
        height = row['h'] * img_size[1]

        # 计算COCO格式坐标
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # 构建标注项
        ann_item = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": class_mapping[row['cls']],
            "bbox": [x_min, y_min, width, height],
            "area": width * height,
            "iscrowd": 0
        }
        coco["annotations"].append(ann_item)
        ann_id += 1

# 保存JSON文件
with open(output_json, 'w') as f:
    json.dump(coco, f, indent=4)
print(f"转换完成！生成文件：{output_json}")