import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import shutil
from joblib import Parallel, delayed

# ===== 配置参数 =====
DATASET_PATH = "/home/a10/slh/yolo/datasets/UAVDT/test/images"  # UAVDT训练集路径
OUTPUT_PATH = "/home/a10/slh/yolo/datasets/UAVDT/cleaned_dataset"  # 清洗后输出路径
TARGET_CLASSES = [0, 1, 2]  # 目标类别ID(车辆、卡车、巴士)
MIN_CONFIDENCE = 0.25  # 检测置信度阈值
MIN_PIXEL_COVERAGE = 0.001  # 最小目标像素覆盖率(0.1%)
BATCH_SIZE = 8  # 批处理大小
NUM_WORKERS = 4  # 并行工作进程数
MODEL_NAME = "yolov8n.pt"  # 使用大模型提高小目标检测精度

# ===== 创建输出目录 =====
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
background_dir = Path(OUTPUT_PATH) / "background"
background_dir.mkdir(exist_ok=True)

# ===== 加载模型 =====
print("🚀 加载检测模型...")
model = YOLO(MODEL_NAME)

# ===== 获取所有图像路径 =====
image_paths = [str(p) for p in Path(DATASET_PATH).glob("*.jpg")]
print(f"📊 共找到 {len(image_paths)} 张训练图像")


# ===== 定义处理函数 =====
def process_image(img_path):
    """处理单张图像，返回是否为背景"""
    img = cv2.imread(img_path)
    if img is None:
        return True, img_path

    # 使用大分辨率检测小目标
    results = model.predict(
        source=img,
        conf=MIN_CONFIDENCE,
        imgsz=1280,  # 提高分辨率检测小目标
        classes=TARGET_CLASSES,
        verbose=False
    )

    # 计算目标覆盖面积
    total_pixels = img.shape[0] * img.shape[1]
    target_pixels = 0

    for result in results:
        for box in result.boxes:
            # 计算边界框面积
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            target_pixels += (x2 - x1) * (y2 - y1)

    # 判断是否为背景
    coverage = target_pixels / total_pixels
    is_background = coverage < MIN_PIXEL_COVERAGE

    return is_background, img_path


# ===== 并行处理所有图像 =====
print("🔍 开始检测背景图像...")
results = Parallel(n_jobs=NUM_WORKERS)(
    delayed(process_image)(path)
    for path in tqdm(image_paths)
)

# ===== 分离背景图像 =====
background_count = 0
for is_background, img_path in results:
    filename = os.path.basename(img_path)

    if is_background:
        # 移动到背景目录
        shutil.copy(img_path, background_dir / filename)
        background_count += 1
    else:
        # 复制到清洗后数据集
        shutil.copy(img_path, Path(OUTPUT_PATH) / filename)

# ===== 输出统计结果 =====
print("\n✅ 处理完成!")
print(f"• 原始图像数量: {len(image_paths)}")
print(f"• 背景图像数量: {background_count}")
print(f"• 保留图像数量: {len(image_paths) - background_count}")
print(f"• 背景比例: {background_count / len(image_paths) * 100:.2f}%")
print(f"• 清洗后数据集已保存至: {OUTPUT_PATH}")
print(f"• 背景图像已保存至: {background_dir}")