from ultralytics import YOLO

# 加载配置
model = YOLO(model='yolov5.yaml', task='detect')

# 训练参数
model.train(
    data='DOTAv1.5.yaml',  # 数据集配置文件
    epochs=300,
    imgsz=512,
    batch=2,  # 可设置为更小的，占据更少的空间。
)

