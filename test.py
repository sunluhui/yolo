from ultralytics import RTDETR
from ultralytics import YOLO
# 加载训练好的模型（必须使用RTDETR类加载RT-DETR模型）
model = RTDETR('runs/detect/train428/weights/best.pt')  # 每次训练完进行测试时，必须修改测试模型的路径
model.val(device=0)  # 明确指定使用GPU 0
# 在测试集上进行评估
metrics = model.val(
    data='VisDrone.yaml',  # 数据集配置文件路径
    split='test',  # 指定使用测试集
)

# 输出评估指标
if metrics.box is not None:
    print(f"mAP50-95: {metrics.box.map:.4f}")        # mAP50-95
    print(f"mAP50: {metrics.box.map50:.4f}")         # mAP50
    print(f"Precision: {metrics.box.p.mean():.4f}")  # 精确率（取平均）
    print(f"Recall: {metrics.box.r.mean():.4f}")     # 召回率（取平均）
    print(f"Per-class mAP50-95: {metrics.box.maps}") # 每个类别的mAP50-95
else:
    print("No detection results!")
