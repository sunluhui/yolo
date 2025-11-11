import torch

from ultralytics import YOLO

# 加载配置xaa
model = YOLO(model='yolov8-p2.yaml', task='detect')
torch.use_deterministic_algorithms(False)
# 训练参数
model.train(
    data='TinyPerson.yaml',  # 数据集配置文件
    epochs=200,
    imgsz=640,
    batch=4,  # 可设置为更小的，占据更少的空间。
)

