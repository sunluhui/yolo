import os
import json
import cv2
from glob import glob


# 配置路径
data_dir = '/home/a10/slh/yolo/datasets//OpenUAVLab_UAVDT/raw'
output_dir = '//home/a10/slh/yolo/datasets/UAVDT_processed'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

# 示例：处理M_attr数据集
image_dir = os.path.join(data_dir, 'M_attr/images')
label_dir = os.path.join(data_dir, 'M_attr/labels')

# 创建COCO格式字典
coco_data = {
    "info": {"description": "UAVDT Dataset"},
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": "uav",
        "supercategory": "object"
    }]
}

# 遍历处理每个图像和标注
image_id = 1
annotation_id = 1
for img_file in sorted(glob(os.path.join(image_dir, '*.jpg'))):
    img_name = os.path.basename(img_file)

    # 处理图像信息
    img_info = {
        "id": image_id,
        "file_name": img_name,
        "width": cv2.imread(img_file).shape[1],
        "height": cv2.imread(img_file).shape[0]
    }
    coco_data["images"].append(img_info)

    # 处理标注文件（假设与图片同名，扩展名为.txt）
    label_file = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            x_center, y_center, width, height, category_id = map(float, parts)

            # 转换坐标到像素格式
            x_min = (x_center - width / 2) * img_info["width"]
            y_min = (y_center - height / 2) * img_info["height"]
            x_max = (x_center + width / 2) * img_info["width"]
            y_max = (y_center + height / 2) * img_info["height"]

            # 添加标注
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1

    image_id += 1

# 保存JSON文件
with open(os.path.join(output_dir, 'annotations', 'instances_M_attr.json'), 'w') as f:
    json.dump(coco_data, f)