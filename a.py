# 检查指定图片的标签文件（将路径替换为实际路径）
import os
from PIL import Image
import matplotlib.pyplot as plt

img_path = "/home/a10/ZH/ultralytics-main/datasets/UAVDT/train/images/image_25215.jpg"
label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

# 可视化检查
img = Image.open(img_path)
plt.imshow(img)
with open(label_path) as f:
    for line in f.readlines():
        cls, x, y, w, h = map(float, line.strip().split())
        print(f"Class {int(cls)} - Center: ({x:.4f}, {y:.4f}), Size: ({w:.4f}, {h:.4f})")
plt.show()