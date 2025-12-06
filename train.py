

import torch

from ultralytics import YOLO
from ultralytics import RTDETR
# 加载配置xaaaaa
model = YOLO(model='yolov6.yaml', task='detect')
torch.use_deterministic_algorithms(False)
# 训练参数
model.train(
    data='VisDrone.yaml',  # 数据集配置文件
    epochs=400,
    imgsz=640,
    batch=8, # 8主数据集, # 可设置为更小的，占据更少的空间。下面的为rtdetr的配置。还得修改训练和测试集的文件头
    optimizer='AdamW',
    lr0=0.0001,
    weight_decay=0.0001,
)

