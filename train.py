from ultralytics import YOLO

# 加载配置
model = YOLO(model='yolov12.yaml', task='detect')

# 训练参数
model.train(
    data='TinyPerson.yaml',  # 数据集配置文件
    epochs=200,
    imgsz=640,
    batch=4,  # 可设置为更小的，占据更少的空间。
)

