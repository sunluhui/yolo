from ultralytics.data.converter import convert_dota_to_yolo_obb

# 基础转换（自动处理class_mapping）
convert_dota_to_yolo_obb(
    dota_path="/home/a10/slh/yolo/datasets/DOTAv1.5/train/labelTxt-v1.5/DOTA-v1.5_train",
    save_dir="/home/a10/slh/yolo/datasets/DOTAv1.5/train/labelTxt-v1.5/DOTA-v1.5_train_yolo"
)