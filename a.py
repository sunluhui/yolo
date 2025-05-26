import os
from glob import glob

# 图片和标签路径
img_dir = "/home/a10/ZH/ultralytics-main/datasets/UAVDT/train/images"
label_dir = "/home/a10/ZH/ultralytics-main/datasets/UAVDT/train/labels"

# 获取所有图片和标签文件名（不带扩展名）
img_files = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]
label_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]

# 找出无标签的图片
no_label_imgs = set(img_files) - set(label_files)

# 删除无标签的图片
for img_name in no_label_imgs:
    img_path = os.path.join(img_dir, img_name + '.jpg')
    os.remove(img_path)
    print(f"Deleted: {img_path}")