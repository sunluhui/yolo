import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 模型配置文件，以相对路径调用，也可以使用绝对路径
model_yaml_path = '/home/a10/slh/ultralytics-main-rtdetr/ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml'
# 数据集配置文件，以相对路径调用，也可以使用绝对路径
data_yaml_path = '/home/a10/slh/ultralytics-main-rtdetr/ultralytics/cfg/datasets/VisDrone.yaml'
if __name__ == '__main__':
    model = RTDETR(model_yaml_path)
    # model.load('rtdetr-l.pt') # 加载预训练权重
    # 训练模型
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=200,
                          batch=8,
                          workers=0,
                          project='runs/RT-DETR/train',
                          name='exp',
                          )

import torch

from ultralytics import YOLO

# 加载配置xaa
#model = YOLO(model='Mamba-YOLOv8-T.yaml', task='detect')
#torch.use_deterministic_algorithms(False)
# 训练参数
#model.train(
#    data='VisDrone.yaml',  # 数据集配置文件
#    epochs=300,
#    imgsz=640,
#    batch=8,  # 可设置为更小的，占据更少的空间。
#)

