import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from config import Config


class YOLODetector:
    def __init__(self, model_path=Config.MODEL_PATH):
        """初始化YOLO检测器"""
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect_image(self, image_path, save_dir=Config.IMAGE_RESULTS_DIR, confidence=Config.CONFIDENCE_THRESHOLD):
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

    def detect_video(self, video_path, save_dir=Config.VIDEO_RESULTS_DIR, confidence=Config.CONFIDENCE_THRESHOLD,
                     progress_callback=None):
        """检测视频并保存结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 生成输出视频路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(save_dir, f"detected_{timestamp}_{base_name}.mp4")

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if progress_callback:
                progress_callback(0, f"无法打开视频文件: {video_path}")
            return None, None

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 检查获取的属性是否有效
        if fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            if progress_callback:
                progress_callback(0, "从视频文件中获取的属性（fps/width/height）无效")
            return None, None

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        detection_data = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 进行预测
                results = self.model(frame, conf=confidence, verbose=False)
                result = results[0]

                # 绘制检测结果
                annotated_frame = result.plot()

                # 确保annotated_frame的尺寸与VideoWriter初始化时一致
                if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                    annotated_frame = cv2.resize(annotated_frame, (width, height))

                # 写入处理后的帧到输出视频
                out.write(annotated_frame)

                # 记录检测信息
                frame_data = {
                    'frame': frame_count,
                    'boxes': result.boxes.data.cpu().numpy() if result.boxes is not None else []
                }
                detection_data.append(frame_data)

                frame_count += 1

                # 更新进度
                if progress_callback and total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    progress_callback(progress)

            # 释放资源
            cap.release()
            out.release()

            # 保存检测结果到文本文件
            txt_output_path = output_path.replace('.mp4', '.txt')
            self._save_video_detection_text(detection_data, txt_output_path)

            return output_path, detection_data

        except Exception as e:
            # 发生异常时，确保释放资源
            cap.release()
            out.release()
            if progress_callback:
                progress_callback(0, f"视频处理过程中发生错误: {str(e)}")
            return None, None
    def detect_camera(self, camera_index=0, save_dir=Config.CAMERA_RESULTS_DIR, confidence=Config.CONFIDENCE_THRESHOLD,
                      max_frames=300, progress_callback=None):
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
        if fps <= 0:
            fps = 30  # 默认帧率

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
            frame_data = {
                'frame': frame_count,
                'boxes': result.boxes.data.cpu().numpy() if result.boxes is not None else []
            }
            detection_data.append(frame_data)

            frame_count += 1

            # 更新进度
            if progress_callback:
                progress = (frame_count / max_frames) * 100
                progress_callback(progress)

        # 释放资源
        cap.release()
        out.release()

        # 保存检测结果到文本文件
        self._save_video_detection_text(detection_data, output_path.replace('.mp4', '.txt'))

        return output_path, detection_data

    def _save_detection_text(self, result, txt_path):
        """保存检测结果到文本文件"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    bbox = box.xywhn[0].cpu().numpy()  # 归一化坐标
                    class_name = self.class_names[class_id]
                    f.write(f"{class_name} {confidence:.2f} {bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}\n")

    def _save_video_detection_text(self, detection_data, txt_path):
        """保存视频检测结果到文本文件"""
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                for frame_data in detection_data:
                    f.write(f"Frame {frame_data['frame']}:\n")
                    if len(frame_data['boxes']) > 0:
                        for box in frame_data['boxes']:
                            class_id = int(box[5])
                            confidence = float(box[4])
                            bbox = box[:4]  # 未归一化坐标
                            class_name = self.class_names[class_id]
                            f.write(
                                f"  {class_name} {confidence:.2f} {bbox[0]:.1f} {bbox[1]:.1f} {bbox[2]:.1f} {bbox[3]:.1f}\n")
                    else:
                        f.write("  No objects detected\n")
        except Exception as e:
            print(f"保存文本结果时出错: {e}")