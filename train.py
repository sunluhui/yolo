from ultralytics import YOLO

# 加载配置
model = YOLO(model='yolov8-CoordAtt.yaml', task='detect')

# 训练参数
model.train(
    data='coco.yaml',  # 数据集配置文件
    epochs=400,
    imgsz=640,
    batch=16,  # 可设置为更小的，占据更少的空间。
)

