import os
import json
from PIL import Image


def json_to_detection_format(json_path, output_dir, img_dir, class_mapping):
    # 加载JSON数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个图像标注
    for img_info in data['images']:
        img_id = img_info['id']
        img_path = os.path.join(img_dir, img_info['file_name'])
        anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]

        # 创建YOLO格式标注文件
        txt_path = os.path.join(output_dir, f"{img_id}.txt")
        with open(txt_path, 'w') as f:
            for ann in anns:
                # 获取类别ID
                cls_id = class_mapping[ann['category_id']]

                # 坐标转换（YOLO格式）
                x_center = (ann['bbox'][0] + ann['bbox'][2] / 2) / img_info['width']
                y_center = (ann['bbox'][1] + ann['bbox'][3] / 2) / img_info['height']
                width = ann['bbox'][2] / img_info['width']
                height = ann['bbox'][3] / img_info['height']

                # 写入文件
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


# 使用示例
class_mapping = {0: 0, 1: 1, 2: 2}  # 类别ID映射
json_to_detection_format(
    json_path='annotations.json',
    output_dir='labels',
    img_dir='images',
    class_mapping=class_mapping
)