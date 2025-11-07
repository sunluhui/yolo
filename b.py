import time
import torch
from ultralytics import YOLO
import glob


def calculate_yolov8_fps(model_path, test_images_dir, img_size=640):
    """
    计算YOLOv8模型的FPS
    """
    # 加载模型
    model = YOLO(model_path)
    model.eval()

    # 获取测试图片
    image_paths = glob.glob(f"{test_images_dir}/*.jpg")[:100]  # 使用100张图片测试

    # Warm-up (重要!)
    print("Warming up GPU...")
    dummy_input = torch.randn(1, 3, img_size, img_size).to(next(model.parameters()).device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # 测量推理时间（包含预处理+推理+后处理）
    print("Measuring FPS...")
    total_time = 0
    num_images = len(image_paths)

    for img_path in image_paths:
        start_time = time.time()

        # 完整推理流程
        results = model.predict(img_path, imgsz=img_size, verbose=False)

        end_time = time.time()
        total_time += (end_time - start_time)

    # 计算FPS
    avg_time_per_image = total_time / num_images
    fps = 1 / avg_time_per_image

    print(f"测试图片数量: {num_images}")
    print(f"总时间: {total_time:.4f}秒")
    print(f"平均每张图片时间: {avg_time_per_image * 1000:.2f}毫秒")
    print(f"FPS: {fps:.2f}")

    return fps


# 使用示例
fps = calculate_yolov8_fps(
    model_path='yolov8n.pt',
    test_images_dir='bus.jpg',
    img_size=640
)