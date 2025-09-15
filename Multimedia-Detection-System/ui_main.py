from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import os
from datetime import datetime
from yolo_detector import YOLODetector
from database import add_detection_record


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, user_id, username):
        super().__init__()
        self.user_id = user_id
        self.username = username
        self.detector = YOLODetector()
        self.current_video_path = None
        self.is_playing = False
        self.is_detecting = False
        self.camera_active = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f'小目标检测系统 - 欢迎 {self.username}使用！')
        self.setGeometry(100, 100, 1400, 900)  # 增大窗口尺寸

        # 设置应用程序样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                font-family: 'Microsoft YaHei', Arial, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background: white;
                border-radius: 5px;
                margin: 5px;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background: #e0e0e0;
                color: #333;
                padding: 10px 20px;
                margin: 5px;
                border-radius: 5px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
            }
            QTabBar::tab:hover {
                background: #cccccc;
            }
            QPushButton {
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                min-width: 100px;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:pressed {
                opacity: 0.8;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QLabel {
                color: #333333;
                font-weight: 500;
            }
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)

        # 中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 标题栏
        header_widget = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QtWidgets.QLabel(f'小目标检测系统 - 欢迎 {self.username}使用！')
        title_font = QtGui.QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50;")

        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # 用户信息和退出按钮
        user_info_label = QtWidgets.QLabel(f"用户ID: {self.user_id}")
        user_info_label.setStyleSheet("color: #7f8c8d;")

        self.logout_btn = QtWidgets.QPushButton('退出系统')
        self.logout_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.logout_btn.clicked.connect(self.close)
        self.logout_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        header_layout.addWidget(user_info_label)
        header_layout.addWidget(self.logout_btn)

        main_layout.addWidget(header_widget)

        # 选项卡
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabs.setMovable(False)

        # 图片检测标签页
        self.image_tab = QtWidgets.QWidget()
        self.setup_image_tab()
        self.tabs.addTab(self.image_tab, "图片检测")

        # 视频检测标签页
        self.video_tab = QtWidgets.QWidget()
        self.setup_video_tab()
        self.tabs.addTab(self.video_tab, "视频检测")

        # 实时摄像头检测标签页
        self.camera_tab = QtWidgets.QWidget()
        self.setup_camera_tab()
        self.tabs.addTab(self.camera_tab, "实时检测")

        main_layout.addWidget(self.tabs)

        # 状态栏
        self.statusBar().showMessage('准备就绪')

        # 设置窗口图标
        self.setWindowIcon(QtGui.QIcon("icon.png"))  # 如果有图标文件的话

    def setup_image_tab(self):
        layout = QtWidgets.QVBoxLayout(self.image_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 控制按钮
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(15)

        self.select_image_btn = QtWidgets.QPushButton('选择图片')
        self.select_image_btn.clicked.connect(self.select_image)
        self.select_image_btn.setStyleSheet("background-color: #3498db;")
        self.select_image_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.select_image_btn)

        self.detect_image_btn = QtWidgets.QPushButton('开始检测')
        self.detect_image_btn.clicked.connect(self.detect_image)
        self.detect_image_btn.setEnabled(False)
        self.detect_image_btn.setStyleSheet("background-color: #4CAF50;")
        self.detect_image_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.detect_image_btn)

        self.save_image_btn = QtWidgets.QPushButton('保存结果')
        self.save_image_btn.clicked.connect(self.save_image_result)
        self.save_image_btn.setEnabled(False)
        self.save_image_btn.setStyleSheet("background-color: #f39c12;")
        self.save_image_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.save_image_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # 图片显示区域
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.setSpacing(20)

        # 原始图片
        original_frame = QtWidgets.QFrame()
        original_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        original_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        original_layout = QtWidgets.QVBoxLayout(original_frame)

        original_title = QtWidgets.QLabel('原始图像')
        original_title.setAlignment(QtCore.Qt.AlignCenter)
        original_title.setStyleSheet("font-weight: bold; background-color: #e8f4fc; padding: 5px; border-radius: 3px;")
        original_layout.addWidget(original_title)

        self.original_image_label = QtWidgets.QLabel()
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_image_label.setMinimumSize(640, 480)
        self.original_image_label.setStyleSheet('border: 1px solid #e0e0e0; background-color: #f8f8f8;')
        self.original_image_label.setText('请选择图片')
        original_layout.addWidget(self.original_image_label)

        image_layout.addWidget(original_frame)

        # 检测结果图片
        result_frame = QtWidgets.QFrame()
        result_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        result_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        result_layout = QtWidgets.QVBoxLayout(result_frame)

        result_title = QtWidgets.QLabel('检测结果')
        result_title.setAlignment(QtCore.Qt.AlignCenter)
        result_title.setStyleSheet("font-weight: bold; background-color: #e8f8e8; padding: 5px; border-radius: 3px;")
        result_layout.addWidget(result_title)

        self.result_image_label = QtWidgets.QLabel()
        self.result_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_image_label.setMinimumSize(640, 480)
        self.result_image_label.setStyleSheet('border: 1px solid #e0e0e0; background-color: #f8f8f8;')
        self.result_image_label.setText('等待检测')
        result_layout.addWidget(self.result_image_label)

        image_layout.addWidget(result_frame)

        layout.addLayout(image_layout)

        # 检测信息
        info_frame = QtWidgets.QFrame()
        info_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        info_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        info_layout = QtWidgets.QVBoxLayout(info_frame)

        info_title = QtWidgets.QLabel('检测信息')
        info_title.setAlignment(QtCore.Qt.AlignCenter)
        info_title.setStyleSheet("font-weight: bold; background-color: #f8f8f8; padding: 5px; border-radius: 3px;")
        info_layout.addWidget(info_title)

        self.image_info_label = QtWidgets.QLabel('检测信息将显示在这里')
        self.image_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.image_info_label.setWordWrap(True)
        self.image_info_label.setStyleSheet("padding: 10px;")
        info_layout.addWidget(self.image_info_label)

        layout.addWidget(info_frame)

        self.image_tab.setLayout(layout)

        self.image_path = None
        self.image_result_path = None
        self.image_detection_data = None

    def setup_video_tab(self):
        layout = QtWidgets.QVBoxLayout(self.video_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 控制按钮
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(15)

        self.select_video_btn = QtWidgets.QPushButton('选择视频')
        self.select_video_btn.clicked.connect(self.select_video)
        self.select_video_btn.setStyleSheet("background-color: #3498db;")
        self.select_video_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.select_video_btn)

        self.play_video_btn = QtWidgets.QPushButton('播放/暂停')
        self.play_video_btn.clicked.connect(self.toggle_video_play)
        self.play_video_btn.setEnabled(False)
        self.play_video_btn.setStyleSheet("background-color: #9b59b6;")
        self.play_video_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.play_video_btn)

        self.detect_video_btn = QtWidgets.QPushButton('开始检测')
        self.detect_video_btn.clicked.connect(self.detect_video)
        self.detect_video_btn.setEnabled(False)
        self.detect_video_btn.setStyleSheet("background-color: #4CAF50;")
        self.detect_video_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.detect_video_btn)

        self.save_video_btn = QtWidgets.QPushButton('保存结果')
        self.save_video_btn.clicked.connect(self.save_video_result)
        self.save_video_btn.setEnabled(False)
        self.save_video_btn.setStyleSheet("background-color: #f39c12;")
        self.save_video_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.save_video_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # 视频显示区域
        video_layout = QtWidgets.QHBoxLayout()
        video_layout.setSpacing(20)

        # 原始视频
        original_frame = QtWidgets.QFrame()
        original_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        original_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        original_layout = QtWidgets.QVBoxLayout(original_frame)

        original_title = QtWidgets.QLabel('原始视频')
        original_title.setAlignment(QtCore.Qt.AlignCenter)
        original_title.setStyleSheet("font-weight: bold; background-color: #e8f4fc; padding: 5px; border-radius: 3px;")
        original_layout.addWidget(original_title)

        self.original_video_label = QtWidgets.QLabel()
        self.original_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_video_label.setMinimumSize(640, 480)
        self.original_video_label.setStyleSheet('border: 1px solid #e0e0e0; background-color: #f8f8f8;')
        self.original_video_label.setText('请选择视频')
        original_layout.addWidget(self.original_video_label)

        video_layout.addWidget(original_frame)

        # 检测结果视频
        result_frame = QtWidgets.QFrame()
        result_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        result_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        result_layout = QtWidgets.QVBoxLayout(result_frame)

        result_title = QtWidgets.QLabel('检测结果')
        result_title.setAlignment(QtCore.Qt.AlignCenter)
        result_title.setStyleSheet("font-weight: bold; background-color: #e8f8e8; padding: 5px; border-radius: 3px;")
        result_layout.addWidget(result_title)

        self.result_video_label = QtWidgets.QLabel()
        self.result_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_video_label.setMinimumSize(640, 480)
        self.result_video_label.setStyleSheet('border: 1px solid #e0e0e0; background-color: #f8f8f8;')
        self.result_video_label.setText('等待检测')
        result_layout.addWidget(self.result_video_label)

        video_layout.addWidget(result_frame)

        layout.addLayout(video_layout)

        # 检测信息
        info_frame = QtWidgets.QFrame()
        info_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        info_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        info_layout = QtWidgets.QVBoxLayout(info_frame)

        info_title = QtWidgets.QLabel('检测信息')
        info_title.setAlignment(QtCore.Qt.AlignCenter)
        info_title.setStyleSheet("font-weight: bold; background-color: #f8f8f8; padding: 5px; border-radius: 3px;")
        info_layout.addWidget(info_title)

        self.video_info_label = QtWidgets.QLabel('检测信息将显示在这里')
        self.video_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet("padding: 10px;")
        info_layout.addWidget(self.video_info_label)

        layout.addWidget(info_frame)

        self.video_tab.setLayout(layout)

        self.video_path = None
        self.video_result_path = None
        self.video_detection_data = None

        # 视频播放定时器
        self.video_timer = QtCore.QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_cap = None

    def setup_camera_tab(self):
        layout = QtWidgets.QVBoxLayout(self.camera_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 控制按钮
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(15)

        self.start_camera_btn = QtWidgets.QPushButton('开启摄像头')
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        self.start_camera_btn.setStyleSheet("background-color: #3498db;")
        self.start_camera_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.start_camera_btn)

        self.detect_camera_btn = QtWidgets.QPushButton('开始检测')
        self.detect_camera_btn.clicked.connect(self.toggle_camera_detection)
        self.detect_camera_btn.setEnabled(False)
        self.detect_camera_btn.setStyleSheet("background-color: #4CAF50;")
        self.detect_camera_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.detect_camera_btn)

        self.save_camera_btn = QtWidgets.QPushButton('保存录像')
        self.save_camera_btn.clicked.connect(self.save_camera_result)
        self.save_camera_btn.setEnabled(False)
        self.save_camera_btn.setStyleSheet("background-color: #f39c12;")
        self.save_camera_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        control_layout.addWidget(self.save_camera_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # 摄像头显示区域
        camera_frame = QtWidgets.QFrame()
        camera_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        camera_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        camera_layout = QtWidgets.QVBoxLayout(camera_frame)

        camera_title = QtWidgets.QLabel('摄像头画面')
        camera_title.setAlignment(QtCore.Qt.AlignCenter)
        camera_title.setStyleSheet("font-weight: bold; background-color: #e8f4fc; padding: 5px; border-radius: 3px;")
        camera_layout.addWidget(camera_title)

        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet('border: 1px solid #e0e0e0; background-color: #f8f8f8;')
        self.camera_label.setText('摄像头未开启')
        camera_layout.addWidget(self.camera_label)

        layout.addWidget(camera_frame)

        # 检测信息
        info_frame = QtWidgets.QFrame()
        info_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        info_frame.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        info_layout = QtWidgets.QVBoxLayout(info_frame)

        info_title = QtWidgets.QLabel('检测信息')
        info_title.setAlignment(QtCore.Qt.AlignCenter)
        info_title.setStyleSheet("font-weight: bold; background-color: #f8f8f8; padding: 5px; border-radius: 3px;")
        info_layout.addWidget(info_title)

        self.camera_info_label = QtWidgets.QLabel('摄像头就绪')
        self.camera_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.camera_info_label.setWordWrap(True)
        self.camera_info_label.setStyleSheet("padding: 10px;")
        info_layout.addWidget(self.camera_info_label)

        layout.addWidget(info_frame)

        self.camera_tab.setLayout(layout)

        self.camera_result_path = None
        self.camera_detection_data = None

        # 摄像头定时器
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.camera_cap = None
        self.is_camera_detecting = False

    def select_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, '选择图片', '',
            '图像文件 (*.jpg *.jpeg *.png *.bmp)'
        )

        if file_path:
            self.image_path = file_path
            pixmap = QtGui.QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.original_image_label.width(),
                self.original_image_label.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.original_image_label.setPixmap(scaled_pixmap)
            self.detect_image_btn.setEnabled(True)
            self.statusBar().showMessage(f'已选择图片: {os.path.basename(file_path)}')

    def detect_image(self):
        if not self.image_path:
            return

        self.statusBar().showMessage('正在检测图片...')
        self.setEnabled(False)
        QtWidgets.QApplication.processEvents()  # 更新UI

        try:
            self.image_result_path, self.image_detection_data = self.detector.detect_image(
                self.image_path
            )

            # 显示检测结果
            if self.image_result_path:
                pixmap = QtGui.QPixmap(self.image_result_path)
                scaled_pixmap = pixmap.scaled(
                    self.result_image_label.width(),
                    self.result_image_label.height(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.result_image_label.setPixmap(scaled_pixmap)

                # 显示检测信息
                if self.image_detection_data.boxes is not None:
                    num_objects = len(self.image_detection_data.boxes)
                    class_counts = {}
                    for box in self.image_detection_data.boxes:
                        class_id = int(box.cls)
                        class_name = self.detector.class_names[class_id]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    info_text = f"检测到 {num_objects} 个对象:\n"
                    for class_name, count in class_counts.items():
                        info_text += f"{class_name}: {count}个\n"

                    self.image_info_label.setText(info_text)
                else:
                    self.image_info_label.setText("未检测到任何对象")

                self.save_image_btn.setEnabled(True)
                self.statusBar().showMessage('图片检测完成')

                # 添加到数据库记录
                add_detection_record(
                    self.user_id,
                    'image',
                    self.image_path,
                    self.image_result_path
                )

        except Exception as e:
            self.statusBar().showMessage(f'检测错误: {str(e)}')
            QtWidgets.QMessageBox.critical(self, '错误', f'检测过程中发生错误: {str(e)}')
        finally:
            self.setEnabled(True)

    def save_image_result(self):
        if not self.image_result_path:
            return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, '保存检测结果',
            f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            'JPEG图像 (*.jpg);;PNG图像 (*.png)'
        )

        if save_path:
            try:
                import shutil
                shutil.copy2(self.image_result_path, save_path)
                self.statusBar().showMessage(f'结果已保存: {os.path.basename(save_path)}')
                QtWidgets.QMessageBox.information(self, '成功', f'图片已保存到: {save_path}')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, '错误', f'保存失败: {str(e)}')

    def select_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, '选择视频', '',
            '视频文件 (*.mp4 *.avi *.mov *.mkv)'
        )

        if file_path:
            self.video_path = file_path
            self.video_cap = cv2.VideoCapture(file_path)

            # 显示第一帧
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.original_video_label.width(),
                    self.original_video_label.height(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.original_video_label.setPixmap(scaled_pixmap)

            self.play_video_btn.setEnabled(True)
            self.detect_video_btn.setEnabled(True)
            self.statusBar().showMessage(f'已选择视频: {os.path.basename(file_path)}')

    def toggle_video_play(self):
        if not self.is_playing:
            self.video_timer.start(30)  # 约33fps
            self.is_playing = True
            self.play_video_btn.setText('暂停')
            self.play_video_btn.setStyleSheet("background-color: #e67e22;")
        else:
            self.video_timer.stop()
            self.is_playing = False
            self.play_video_btn.setText('播放')
            self.play_video_btn.setStyleSheet("background-color: #9b59b6;")

    def update_video_frame(self):
        if self.video_cap is None:
            return

        ret, frame = self.video_cap.read()
        if not ret:
            self.video_timer.stop()
            self.is_playing = False
            self.play_video_btn.setText('播放')
            self.play_video_btn.setStyleSheet("background-color: #9b59b6;")
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.original_video_label.width(),
            self.original_video_label.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.original_video_label.setPixmap(scaled_pixmap)

    def detect_video(self):
        if not self.video_path:
            return

        self.statusBar().showMessage('正在检测视频，这可能需要一些时间...')
        self.setEnabled(False)  # 禁用界面防止重复操作
        QtWidgets.QApplication.processEvents()  # 更新UI

        # 在子线程中执行检测
        from threading import Thread

        def video_detection_thread():
            try:
                self.video_result_path, self.video_detection_data = self.detector.detect_video(
                    self.video_path
                )

                # 回到主线程更新UI
                QtCore.QMetaObject.invokeMethod(self, '_video_detection_finished',
                                                QtCore.Qt.QueuedConnection)
            except Exception as e:
                error_msg = str(e)
                QtCore.QMetaObject.invokeMethod(self, '_video_detection_error',
                                                QtCore.Qt.QueuedConnection,
                                                QtCore.Q_ARG(str, error_msg))

        self.detection_thread = Thread(target=video_detection_thread)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def _video_detection_finished(self):
        self.setEnabled(True)

        if self.video_result_path:
            # 显示检测完成的视频的第一帧
            cap = cv2.VideoCapture(self.video_result_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.result_video_label.width(),
                    self.result_video_label.height(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.result_video_label.setPixmap(scaled_pixmap)
            cap.release()

            # 显示检测信息
            if self.video_detection_data:
                total_objects = sum(len(frame['boxes']) for frame in self.video_detection_data)
                self.video_info_label.setText(f"视频检测完成，共检测到 {total_objects} 个对象")

            self.save_video_btn.setEnabled(True)
            self.statusBar().showMessage('视频检测完成')

            # 添加到数据库记录
            add_detection_record(
                self.user_id,
                'video',
                self.video_path,
                self.video_result_path
            )

    def _video_detection_error(self, error_msg):
        self.setEnabled(True)
        self.statusBar().showMessage(f'视频检测错误: {error_msg}')
        QtWidgets.QMessageBox.critical(self, '错误', f'视频检测过程中发生错误: {error_msg}')

    def save_video_result(self):
        if not self.video_result_path:
            return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, '保存检测结果视频',
            f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            'MP4视频 (*.mp4);;AVI视频 (*.avi)'
        )

        if save_path:
            try:
                import shutil
                shutil.copy2(self.video_result_path, save_path)
                self.statusBar().showMessage(f'结果视频已保存: {os.path.basename(save_path)}')
                QtWidgets.QMessageBox.information(self, '成功', f'视频已保存到: {save_path}')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, '错误', f'保存失败: {str(e)}')

    def toggle_camera(self):
        if not self.camera_active:
            # 开启摄像头
            self.camera_cap = cv2.VideoCapture(0)
            if not self.camera_cap.isOpened():
                QtWidgets.QMessageBox.warning(self, '错误', '无法打开摄像头')
                return

            self.camera_timer.start(30)  # 约33fps
            self.camera_active = True
            self.start_camera_btn.setText('关闭摄像头')
            self.start_camera_btn.setStyleSheet("background-color: #e74c3c;")
            self.detect_camera_btn.setEnabled(True)
            self.statusBar().showMessage('摄像头已开启')
        else:
            # 关闭摄像头
            self.camera_timer.stop()
            if self.camera_cap:
                self.camera_cap.release()
                self.camera_cap = None
            self.camera_active = False
            self.start_camera_btn.setText('开启摄像头')
            self.start_camera_btn.setStyleSheet("background-color: #3498db;")
            self.detect_camera_btn.setEnabled(False)
            self.detect_camera_btn.setText('开始检测')
            self.detect_camera_btn.setStyleSheet("background-color: #4CAF50;")
            self.is_camera_detecting = False
            self.camera_label.clear()
            self.camera_label.setText('摄像头画面')
            self.statusBar().showMessage('摄像头已关闭')

    def update_camera_frame(self):
        if self.camera_cap is None:
            return

        ret, frame = self.camera_cap.read()
        if not ret:
            return

        if self.is_camera_detecting:
            # 进行实时检测
            results = self.detector.model(frame, conf=0.5, verbose=False)
            result = results[0]
            frame = result.plot()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)

    def toggle_camera_detection(self):
        self.is_camera_detecting = not self.is_camera_detecting
        if self.is_camera_detecting:
            self.detect_camera_btn.setText('停止检测')
            self.detect_camera_btn.setStyleSheet("background-color: #e74c3c;")
            self.save_camera_btn.setEnabled(True)
            self.statusBar().showMessage('实时检测已开启')
        else:
            self.detect_camera_btn.setText('开始检测')
            self.detect_camera_btn.setStyleSheet("background-color: #4CAF50;")
            self.statusBar().showMessage('实时检测已关闭')

    def save_camera_result(self):
        if not self.camera_active:
            return

        self.statusBar().showMessage('正在保存摄像头录像...')
        QtWidgets.QApplication.processEvents()  # 更新UI

        # 停止摄像头临时停止录制
        self.camera_timer.stop()

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, '保存摄像头录像',
            f"camera_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            'MP4视频 (*.mp4);;AVI视频 (*.avi)'
        )

        if save_path:
            try:
                # 录制5秒视频
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, 20.0, (640, 480))

                # 录制5秒
                for _ in range(100):  # 20fps * 5秒 = 100帧
                    ret, frame = self.camera_cap.read()
                    if ret:
                        if self.is_camera_detecting:
                            results = self.detector.model(frame, conf=0.5, verbose=False)
                            result = results[0]
                            frame = result.plot()
                        out.write(frame)

                out.release()
                self.statusBar().showMessage(f'摄像头录像已保存: {os.path.basename(save_path)}')
                QtWidgets.QMessageBox.information(self, '成功', f'录像已保存到: {save_path}')

                # 添加到数据库记录
                add_detection_record(
                    self.user_id,
                    'camera',
                    '实时摄像头',
                    save_path
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, '错误', f'保存失败: {str(e)}')

        # 重新开启摄像头
        self.camera_timer.start(30)

    def closeEvent(self, event):
        # 清理资源
        if self.video_cap:
            self.video_cap.release()
        if self.camera_cap:
            self.camera_cap.release()
        if self.video_timer.isActive():
            self.video_timer.stop()
        if self.camera_timer.isActive():
            self.camera_timer.stop()

        # 创建QMessageBox实例
        msg_box = QtWidgets.QMessageBox(self)  # 指定父窗口为self，确保对话框居中显示
        msg_box.setWindowTitle('确认退出')
        msg_box.setText('确定要退出系统吗？')
        msg_box.setIcon(QtWidgets.QMessageBox.Question)

        # 添加自定义按钮
        confirm_button = msg_box.addButton('确认', QtWidgets.QMessageBox.YesRole)  # 使用YesRole
        cancel_button = msg_box.addButton('取消', QtWidgets.QMessageBox.NoRole)  # 使用NoRole

        # 设置按钮样式表，鼠标悬停时变红
        button_style = """
            QPushButton {
                color: black;
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #ff0000;  /* 鼠标悬停时变为红色 */
                color: white;               /* 文字颜色变为白色，提高对比度 */
            }
            QPushButton:pressed {
                background-color: #cc0000;  /* 按钮按下时变为深红色 */
                color: white;
            }
        """

        confirm_button.setStyleSheet(button_style)
        cancel_button.setStyleSheet(button_style)

        # 设置默认按钮（默认选中取消）
        msg_box.setDefaultButton(cancel_button)

        # 显示消息框并等待用户响应
        msg_box.exec_()  # 使用exec_()确保对话框模态显示

        # 判断用户点击了哪个按钮
        if msg_box.clickedButton() == confirm_button:
            print('用户点击了确认，退出系统')
            event.accept()  # 接受关闭事件，退出程序
        else:
            print('用户点击了取消，继续使用系统')
            event.ignore()  # 忽略关闭事件，继续运行程序