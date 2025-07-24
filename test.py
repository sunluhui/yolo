from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train287/weights/best.pt')  # 每次训练完进行测试时，必须修改测试模型的路径

# 在测试集上进行评估
metrics = model.val(
    data='coco.yaml',  # 数据集配置文件路径
    split='val',  # 指定使用测试集
)

# 输出评估指标
print(f"mAP50-95: {metrics.box.map:.4f}")    # mAP50-95
print(f"mAP50: {metrics.box.map50:.4f}")     # mAP50
print(f"Precision: {metrics.box.p:.4f}")     # 精确率
print(f"Recall: {metrics.box.r:.4f}")        # 召回率
print(f"Per-class mAP50-95: {metrics.box.maps}")  # 每个类别的mAP50-95
