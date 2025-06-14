from ultralytics import YOLO

# 加载配置
model = YOLO(model='rtdetr-l.yaml', task='detect')

# 训练参数
model.train(
    data='VisDrone.yaml',  # 数据集配置文件
    epochs=300,
    imgsz=640,
    batch=8,  # 可设置为更小的，占据更少的空间。
)

