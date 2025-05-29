import os

# 配置路径
img_dir = "/home/a10/slh/yolo/datasets/UAVDT/cleaned_dataset"  # 清洗后的图片目录
label_dir = "/home/a10/slh/yolo/datasets/UAVDT/test/labels"  # 标签目录

# 获取清洗后保留的图片名（不带扩展名）
valid_images = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}

# 遍历标签目录并删除无效标签
for label_file in os.listdir(label_dir):
    label_name = os.path.splitext(label_file)[0]  # 去掉.txt后缀

    if label_name not in valid_images:
        os.remove(os.path.join(label_dir, label_file))
        print(f"Deleted label: {label_file}")