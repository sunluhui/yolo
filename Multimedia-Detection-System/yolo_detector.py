import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime


class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        """初始化YOLO检测器"""
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect_image(self, image_path, save_dir='results/images', confidence=0.5):
        """检测单张图片并保存结果"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 进行预测
        results = self.model(image_path, conf=confidence)

        # 处理结果
        result = results[0]
        output_path = os.path.join(
            save_dir,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(image_path)}"
        )

        # 保存带标注的图像
        annotated_image = result.plot()
        cv2.imwrite(output_path, annotated_image)

        # 保存检测结果到文本文件
        self._save_detection_text(result, output_path.replace('.jpg', '.txt'))

        return output_path, result

    def detect_video(self, video_path, save_dir='results/videos', confidence=0.5):
        """检测视频并保存结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 生成输出视频路径
        output_path = os.path.join(
            save_dir,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(video_path)}"
        )

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        detection_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 进行预测
            results = self.model(frame, conf=confidence, verbose=False)
            result = results[0]

            # 绘制检测结果
            annotated_frame = result.plot()
            out.write(annotated_frame)

            # 记录检测信息
            detection_data.append({
                'frame': frame_count,
                'boxes': result.boxes.data.cpu().numpy() if result.boxes is not None else []
            })

            frame_count += 1

        # 释放资源
        cap.release()
        out.release()

        # 保存检测结果到文本文件
        self._save_video_detection_text(detection_data, output_path.replace('.mp4', '.txt'))

        return output_path, detection_data

    def detect_camera(self, camera_index=0, save_dir='results/camera', confidence=0.5, max_frames=100):
        """从摄像头实时检测并保存结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 生成输出视频路径
        output_path = os.path.join(
            save_dir,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_camera.mp4"
        )

        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None, None

        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        detection_data = []

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 进行预测
            results = self.model(frame, conf=confidence, verbose=False)
            result = results[0]

            # 绘制检测结果
            annotated_frame = result.plot()
            out.write(annotated_frame)

            # 记录检测信息
            detection_data.append({
                'frame': frame_count,
                'boxes': result.boxes.data.cpu().numpy() if result.boxes is not None else []
            })

            frame_count += 1

        # 释放资源
        cap.release()
        out.release()

        # 保存检测结果到文本文件
        self._save_video_detection_text(detection_data, output_path.replace('.mp4', '.txt'))

        return output_path, detection_data

    def _save_detection_text(self, result, txt_path):
        """保存检测结果到文本文件:cite[2]:cite[7]"""
        with open(txt_path, 'w') as f:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    bbox = box.xywhn[0].cpu().numpy()  # 归一化坐标
                    f.write(f"{class_id} {confidence} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    def _save_video_detection_text(self, detection_data, txt_path):
        """保存视频检测结果到文本文件"""
        with open(txt_path, 'w') as f:
            for frame_data in detection_data:
                f.write(f"Frame {frame_data['frame']}:\n")
                for box in frame_data['boxes']:
                    class_id = int(box[5])
                    confidence = float(box[4])
                    bbox = box[:4]  # 未归一化坐标
                    f.write(f"  {class_id} {confidence} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")