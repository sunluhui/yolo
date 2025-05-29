import os
import cv2
from pathlib import Path

# 设置数据集路径
dataset_path = "/home/a10/slh/yolo/datasets/OpenDataLab_UAVDT/raw/UAV-benchmark-M"
output_path = os.path.join(dataset_path, "labels")

# 创建输出目录(如果不存在)
os.makedirs(output_path, exist_ok=True)

# 类别映射，根据UAV-benchmark-M的实际类别修改
class_mapping = {
    1: 0,  # 示例: 类别0映射到YOLO的类别0
    2: 1,
    3: 2   # 示例: 类别1映射到YOLO的类别1
    # 添加更多类别...
}

# 遍历所有序列文件夹
sequence_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith("M") and len(d) == 5])
for seq_dir in sequence_dirs:
    seq_path = os.path.join(dataset_path, seq_dir)
    img_dir = os.path.join(seq_path, "images")  # 假设图像在images子文件夹中

    # 获取所有图像文件
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img_name = os.path.splitext(img_file)[0]

        # 读取图像获取尺寸
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # 获取对应的标签文件路径
        label_file = os.path.join(seq_path, "labels", f"{img_name}.txt")  # 假设标签在labels子文件夹中
        # 或者从其他格式的标注文件中读取(如VOC XML)

        # 创建YOLO格式的标签文件
        yolo_label_path = os.path.join(output_path, f"{seq_dir}_{img_name}.txt")

        with open(yolo_label_path, 'w') as f:
            # 如果是直接读取YOLO格式的标签文件
            if os.path.exists(label_file):
                with open(label_file, 'r') as lf:
                    lines = lf.readlines()
                    for line in lines:
                        class_id, x_center, y_center, bbox_width, bbox_height = line.strip().split()
                        class_id = int(class_id)

                        # 将坐标转换为相对于图像宽高的归一化值
                        x_center = float(x_center)
                        y_center = float(y_center)
                        bbox_width = float(bbox_width)
                        bbox_height = float(bbox_height)

                        # 写入YOLO格式
                        f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
            else:
                # 如果需要从其他格式转换，请在此处添加代码
                pass