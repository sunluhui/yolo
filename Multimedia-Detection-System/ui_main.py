from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import os
from datetime import datetime
from threading import Thread


class MainWindow(QtWidgets.QMainWindow):
    # 定义信号
    progress_updated = QtCore.pyqtSignal(int)
    error_occurred = QtCore.pyqtSignal(str)
    detection_finished = QtCore.pyqtSignal()

    def __init__(self, user_id, username, db_manager, detector):
        super().__init__()
        self.user_id = user_id
        self.username = username
        self.db_manager = db_manager
        self.detector = detector
        self.current_video_path = None
        self.is_playing = False
        self.is_detecting = False
        self.camera_active = False

        # 连接信号和槽
        self.progress_updated.connect(self.update_video_progress)
        self.error_occurred.connect(self._video_detection_error)
        self.detection_finished.connect(self._video_detection_finished)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f'小目标检测系统 - 欢迎 {self.username}使用！')
        self.setGeometry(100, 100, 1300, 850)  # 增加窗口尺寸
        self.setStyleSheet(self.get_main_window_style())

        # 中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 选项卡
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(self.get_tab_style())

        # 图片检测标签页
        self.image_tab = QtWidgets.QWidget()
        self.setup_image_tab()
        self.tabs.addTab(self.image_tab, "📷 图片检测")

        # 视频检测标签页
        self.video_tab = QtWidgets.QWidget()
        self.setup_video_tab()
        self.tabs.addTab(self.video_tab, "🎬 视频检测")

        # 实时摄像头检测标签页
        self.camera_tab = QtWidgets.QWidget()
        self.setup_camera_tab()
        self.tabs.addTab(self.camera_tab, "📹 实时检测")

        # 历史记录标签页
        self.history_tab = QtWidgets.QWidget()
        self.setup_history_tab()
        self.tabs.addTab(self.history_tab, "📊 历史记录")

        main_layout.addWidget(self.tabs)

        # 状态栏
        self.statusBar().showMessage('就绪')
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2c3e50;
                color: white;
                font-weight: bold;
            }
        """)

    def get_main_window_style(self):
        """返回主窗口样式表"""
        return """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ecf0f1, stop:1 #bdc3c7);
                font-family: "Microsoft YaHei";
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit, QComboBox {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
                selection-background-color: #3498db;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 2px solid #3498db;
                background-color: #eaf2f8;
            }
            QLineEdit:hover, QComboBox:hover {
                border: 2px solid #3498db;
            }
            QTableWidget {
                gridline-color: #bdc3c7;
                background-color: white;
                alternate-background-color: #f5f5f5;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
            QProgressDialog {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
            }
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
            }
        """

    def get_tab_style(self):
        """返回选项卡样式表"""
        return """
            QTabWidget::pane {
                border: 2px solid #bdc3c7;
                background-color: #ecf0f1;
                border-radius: 8px;
                margin-top: 10px;
            }
            QTabBar::tab {
                background: #95a5a6;
                color: white;
                padding: 12px 24px;
                border: 1px solid #7f8c8d;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                border-bottom: 2px solid #3498db;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34495e, stop:1 #2c3e50);
            }
        """

    def get_button_style(self, color1, color2):
        """返回按钮样式表"""
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {color1}, stop:1 {color2});
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5dade2, stop:1 #3498db);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2c3e50, stop:1 #34495e);
                padding: 9px;
            }}
            QPushButton:disabled {{
                background: #bdc3c7;
                color: #7f8c8d;
            }}
        """

    def setup_image_tab(self):
        layout = QtWidgets.QVBoxLayout(self.image_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 控制按钮
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(10)

        self.select_image_btn = QtWidgets.QPushButton('📁 选择图片')
        self.select_image_btn.clicked.connect(self.select_image)
        self.select_image_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.select_image_btn.setStyleSheet(self.get_button_style("#3498db", "#2980b9"))
        control_layout.addWidget(self.select_image_btn)

        self.detect_image_btn = QtWidgets.QPushButton('🔍 开始检测')
        self.detect_image_btn.clicked.connect(self.detect_image)
        self.detect_image_btn.setEnabled(False)
        self.detect_image_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.detect_image_btn.setStyleSheet(self.get_button_style("#2ecc71", "#27ae60"))
        control_layout.addWidget(self.detect_image_btn)

        self.save_image_btn = QtWidgets.QPushButton('💾 保存结果')
        self.save_image_btn.clicked.connect(self.save_image_result)
        self.save_image_btn.setEnabled(False)
        self.save_image_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.save_image_btn.setStyleSheet(self.get_button_style("#9b59b6", "#8e44ad"))
        control_layout.addWidget(self.save_image_btn)

        layout.addLayout(control_layout)

        # 图片显示区域
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.setSpacing(15)

        # 原始图片
        image_group1 = QtWidgets.QGroupBox("原始图像")
        image_group1.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        group_layout1 = QtWidgets.QVBoxLayout()
        self.original_image_label = QtWidgets.QLabel()
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_image_label.setMinimumSize(640, 480)
        self.original_image_label.setStyleSheet("""
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            background-color: #f8f9fa;
        """)
        self.original_image_label.setText('请选择图片')
        group_layout1.addWidget(self.original_image_label)
        image_group1.setLayout(group_layout1)
        image_layout.addWidget(image_group1)

        # 检测结果图片
        image_group2 = QtWidgets.QGroupBox("检测结果")
        image_group2.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        group_layout2 = QtWidgets.QVBoxLayout()
        self.result_image_label = QtWidgets.QLabel()
        self.result_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_image_label.setMinimumSize(640, 480)
        self.result_image_label.setStyleSheet("""
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            background-color: #f8f9fa;
        """)
        self.result_image_label.setText('等待检测')
        group_layout2.addWidget(self.result_image_label)
        image_group2.setLayout(group_layout2)
        image_layout.addWidget(image_group2)

        layout.addLayout(image_layout)

        # 检测信息
        info_group = QtWidgets.QGroupBox("检测信息")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        info_layout = QtWidgets.QVBoxLayout()
        self.image_info_label = QtWidgets.QLabel('检测信息将显示在这里！')
        self.image_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.image_info_label.setWordWrap(True)
        self.image_info_label.setStyleSheet("""
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #bdc3c7;
        """)
        info_layout.addWidget(self.image_info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

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
        control_layout.setSpacing(10)

        self.select_video_btn = QtWidgets.QPushButton('📁 选择视频')
        self.select_video_btn.clicked.connect(self.select_video)
        self.select_video_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.select_video_btn.setStyleSheet(self.get_button_style("#3498db", "#2980b9"))
        control_layout.addWidget(self.select_video_btn)

        self.play_video_btn = QtWidgets.QPushButton('⏯️ 播放/暂停')
        self.play_video_btn.clicked.connect(self.toggle_video_play)
        self.play_video_btn.setEnabled(False)
        self.play_video_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.play_video_btn.setStyleSheet(self.get_button_style("#f39c12", "#e67e22"))
        control_layout.addWidget(self.play_video_btn)

        self.detect_video_btn = QtWidgets.QPushButton('🔍 开始检测')
        self.detect_video_btn.clicked.connect(self.detect_video)
        self.detect_video_btn.setEnabled(False)
        self.detect_video_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.detect_video_btn.setStyleSheet(self.get_button_style("#2ecc71", "#27ae60"))
        control_layout.addWidget(self.detect_video_btn)

        self.save_video_btn = QtWidgets.QPushButton('💾 保存结果')
        self.save_video_btn.clicked.connect(self.save_video_result)
        self.save_video_btn.setEnabled(False)
        self.save_video_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.save_video_btn.setStyleSheet(self.get_button_style("#9b59b6", "#8e44ad"))
        control_layout.addWidget(self.save_video_btn)

        layout.addLayout(control_layout)

        # 视频显示区域
        video_layout = QtWidgets.QHBoxLayout()
        video_layout.setSpacing(15)

        # 原始视频
        video_group1 = QtWidgets.QGroupBox("原始视频")
        video_group1.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        group_layout1 = QtWidgets.QVBoxLayout()
        self.original_video_label = QtWidgets.QLabel()
        self.original_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_video_label.setMinimumSize(640, 480)
        self.original_video_label.setStyleSheet("""
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            background-color: #f8f9fa;
        """)
        self.original_video_label.setText('请选择视频')
        group_layout1.addWidget(self.original_video_label)
        video_group1.setLayout(group_layout1)
        video_layout.addWidget(video_group1)

        # 检测结果视频
        video_group2 = QtWidgets.QGroupBox("检测结果")
        video_group2.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        group_layout2 = QtWidgets.QVBoxLayout()
        self.result_video_label = QtWidgets.QLabel()
        self.result_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_video_label.setMinimumSize(640, 480)
        self.result_video_label.setStyleSheet("""
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            background-color: #f8f9fa;
        """)
        self.result_video_label.setText('等待检测')
        group_layout2.addWidget(self.result_video_label)
        video_group2.setLayout(group_layout2)
        video_layout.addWidget(video_group2)

        layout.addLayout(video_layout)

        # 检测信息
        info_group = QtWidgets.QGroupBox("检测信息")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        info_layout = QtWidgets.QVBoxLayout()
        self.video_info_label = QtWidgets.QLabel('检测信息将显示在这里！')
        self.video_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet("""
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #bdc3c7;
        """)
        info_layout.addWidget(self.video_info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        self.video_tab.setLayout(layout)

        self.video_path = None
        self.video_result_path = None
        self.video_detection_data = None

        # 视频播放定时器
        self.video_timer = QtCore.QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_cap = None

        # 进度对话框
        self.progress_dialog = None

    def setup_camera_tab(self):
        layout = QtWidgets.QVBoxLayout(self.camera_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 控制按钮
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(10)

        self.start_camera_btn = QtWidgets.QPushButton('📹 开启摄像头')
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        self.start_camera_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.start_camera_btn.setStyleSheet(self.get_button_style("#3498db", "#2980b9"))
        control_layout.addWidget(self.start_camera_btn)

        self.detect_camera_btn = QtWidgets.QPushButton('🔍 开始检测')
        self.detect_camera_btn.clicked.connect(self.toggle_camera_detection)
        self.detect_camera_btn.setEnabled(False)
        self.detect_camera_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.detect_camera_btn.setStyleSheet(self.get_button_style("#2ecc71", "#27ae60"))
        control_layout.addWidget(self.detect_camera_btn)

        self.save_camera_btn = QtWidgets.QPushButton('💾 保存录像')
        self.save_camera_btn.clicked.connect(self.save_camera_result)
        self.save_camera_btn.setEnabled(False)
        self.save_camera_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.save_camera_btn.setStyleSheet(self.get_button_style("#9b59b6", "#8e44ad"))
        control_layout.addWidget(self.save_camera_btn)

        layout.addLayout(control_layout)

        # 摄像头显示区域
        camera_layout = QtWidgets.QHBoxLayout()

        # 实时摄像头
        camera_group = QtWidgets.QGroupBox("摄像头画面")
        camera_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        group_layout = QtWidgets.QVBoxLayout()
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            background-color: #f8f9fa;
        """)
        self.camera_label.setText('摄像头未开启')
        group_layout.addWidget(self.camera_label)
        camera_group.setLayout(group_layout)
        camera_layout.addWidget(camera_group)

        layout.addLayout(camera_layout)

        # 检测信息
        info_group = QtWidgets.QGroupBox("检测信息")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        info_layout = QtWidgets.QVBoxLayout()
        self.camera_info_label = QtWidgets.QLabel('摄像头就绪')
        self.camera_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.camera_info_label.setWordWrap(True)
        self.camera_info_label.setStyleSheet("""
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #bdc3c7;
        """)
        info_layout.addWidget(self.camera_info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        self.camera_tab.setLayout(layout)

        self.camera_result_path = None
        self.camera_detection_data = None

        # 摄像头定时器
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.camera_cap = None
        self.is_camera_detecting = False
        self.is_recording = False
        self.video_writer = None

    def setup_history_tab(self):
        layout = QtWidgets.QVBoxLayout(self.history_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QtWidgets.QLabel('📊 检测历史记录')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # 历史记录表格
        self.history_table = QtWidgets.QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(['检测类型', '源文件', '结果文件', '检测时间'])
        self.history_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.history_table.setAlternatingRowColors(True)
        layout.addWidget(self.history_table)

        # 刷新按钮
        self.refresh_btn = QtWidgets.QPushButton('🔄 刷新记录')
        self.refresh_btn.clicked.connect(self.load_history)
        self.refresh_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.refresh_btn.setStyleSheet(self.get_button_style("#3498db", "#2980b9"))
        layout.addWidget(self.refresh_btn)

        self.history_tab.setLayout(layout)

        # 加载历史记录
        self.load_history()

    def load_history(self):
        """加载用户的检测历史记录"""
        records = self.db_manager.get_detection_records(self.user_id)

        self.history_table.setRowCount(len(records))
        for row, record in enumerate(records):
            detection_type, source_path, result_path, detection_time = record

            # 只显示文件名而非完整路径
            source_file = os.path.basename(source_path) if source_path else "实时摄像头"
            result_file = os.path.basename(result_path) if result_path else "无"

            self.history_table.setItem(row, 0, QtWidgets.QTableWidgetItem(detection_type))
            self.history_table.setItem(row, 1, QtWidgets.QTableWidgetItem(source_file))
            self.history_table.setItem(row, 2, QtWidgets.QTableWidgetItem(result_file))

            # 确保时间显示格式正确
            if isinstance(detection_time, datetime):
                time_str = detection_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = str(detection_time)

            self.history_table.setItem(row, 3, QtWidgets.QTableWidgetItem(time_str))

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
                QtCore.Qt.KeepAspectRatio
            )
            self.original_image_label.setPixmap(scaled_pixmap)
            self.detect_image_btn.setEnabled(True)
            self.statusBar().showMessage(f'已选择图片: {os.path.basename(file_path)}')

    def detect_image(self):
        if not self.image_path:
            return

        self.statusBar().showMessage('正在检测图片...')
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
                    QtCore.Qt.KeepAspectRatio
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
                self.db_manager.add_detection_record(
                    self.user_id,
                    'image',
                    self.image_path,
                    self.image_result_path
                )

                # 刷新历史记录
                self.load_history()

        except Exception as e:
            self.statusBar().showMessage(f'检测错误: {str(e)}')
            QtWidgets.QMessageBox.critical(self, '错误', f'检测过程中发生错误: {str(e)}')

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
                    QtCore.Qt.KeepAspectRatio
                )
                self.original_video_label.setPixmap(scaled_pixmap)

            self.play_video_btn.setEnabled(True)
            self.detect_video_btn.setEnabled(True)
            self.statusBar().showMessage(f'已选择视频: {os.path.basename(file_path)}')

    def toggle_video_play(self):
        if not self.is_playing:
            self.video_timer.start(30)  # 约33fps
            self.is_playing = True
            self.play_video_btn.setText('⏸️ 暂停')
        else:
            self.video_timer.stop()
            self.is_playing = False
            self.play_video_btn.setText('⏯️ 播放')

    def update_video_frame(self):
        if self.video_cap is None:
            return

        ret, frame = self.video_cap.read()
        if not ret:
            self.video_timer.stop()
            self.is_playing = False
            self.play_video_btn.setText('⏯️ 播放')
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
            QtCore.Qt.KeepAspectRatio
        )
        self.original_video_label.setPixmap(scaled_pixmap)

    def detect_video(self):
        if not self.video_path:
            return

        self.statusBar().showMessage('正在检测视频，这可能需要一些时间...')
        self.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        # 创建进度对话框
        self.progress_dialog = QtWidgets.QProgressDialog("处理视频中...", "取消", 0, 200, self)
        self.progress_dialog.setWindowTitle("视频检测")
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.resize(300, 250)  # 宽 600 px，高 250 px，按需调
        self.progress_dialog.setStyleSheet("""
            QProgressDialog {
                background-color: white;
                border: 8px solid #3498db;
                border-radius: 8px;
            }
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
            }
        """)
        self.progress_dialog.canceled.connect(self.cancel_video_detection)
        self.progress_dialog.show()

        # 在子线程中执行检测
        def video_detection_thread():
            try:
                # 进度回调函数
                def progress_callback(progress, error_msg=None):
                    if error_msg is not None:
                        # 使用信号发射错误信息
                        self.error_occurred.emit(error_msg)
                    else:
                        # 使用信号发射进度信息
                        self.progress_updated.emit(progress)

                self.video_result_path, self.video_detection_data = self.detector.detect_video(
                    self.video_path, progress_callback=progress_callback
                )

                # 回到主线程更新UI
                self.detection_finished.emit()

            except Exception as e:
                error_msg = str(e)
                # 使用信号发射错误信息
                self.error_occurred.emit(error_msg)

        self.detection_thread = Thread(target=video_detection_thread)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def cancel_video_detection(self):
        # 设置取消标志，需要在Detector类中添加相应的取消机制
        self.statusBar().showMessage('视频检测已取消')
        self.setEnabled(True)
        self.progress_dialog.close()

    def update_video_progress(self, progress_value):
        """更新视频处理进度
        Args:
            progress_value (int): 进度值 (0-100)
        """
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(progress_value)
        self.statusBar().showMessage(f'视频处理进度: {progress_value}%')

    def _video_detection_error(self, error_message):
        """处理视频检测错误
        Args:
            error_message (str): 错误信息
        """
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
        self.setEnabled(True)
        self.statusBar().showMessage(f'视频检测错误: {error_message}')
        QtWidgets.QMessageBox.critical(self, '错误', f'视频检测过程中发生错误: {error_message}')

    def _video_detection_finished(self):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
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
                    QtCore.Qt.KeepAspectRatio
                )
                self.result_video_label.setPixmap(scaled_pixmap)
            cap.release()

            # 显示检测信息
            if self.video_detection_data:
                total_objects = sum(len(frame_data['boxes']) for frame_data in self.video_detection_data)
                self.video_info_label.setText(f"视频检测完成，共检测到 {total_objects} 个对象")

            self.save_video_btn.setEnabled(True)
            self.statusBar().showMessage('视频检测完成')

            # 添加到数据库记录
            self.db_manager.add_detection_record(
                self.user_id,
                'video',
                self.video_path,
                self.video_result_path
            )

            # 刷新历史记录
            self.load_history()

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
            self.start_camera_btn.setText('📹 关闭摄像头')
            self.detect_camera_btn.setEnabled(True)
            self.statusBar().showMessage('摄像头已开启')

            # 初始化录像相关变量
            self.is_recording = False
            self.video_writer = None
        else:
            # 关闭摄像头
            self.camera_timer.stop()
            if self.camera_cap:
                self.camera_cap.release()
                self.camera_cap = None

            # 如果正在录制，停止录制
            if self.is_recording and self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.is_recording = False

            self.camera_active = False
            self.start_camera_btn.setText('📹 开启摄像头')
            self.detect_camera_btn.setEnabled(False)
            self.detect_camera_btn.setText('🔍 开始检测')
            self.save_camera_btn.setEnabled(False)
            self.save_camera_btn.setText('💾 保存录像')
            self.is_camera_detecting = False
            self.camera_label.clear()
            self.camera_label.setText('摄像头未开启')
            self.statusBar().showMessage('摄像头已关闭')

    def update_camera_frame(self):
        if self.camera_cap is None:
            return

        ret, frame = self.camera_cap.read()
        if not ret:
            return

        # 如果正在检测，执行目标检测
        if self.is_camera_detecting:
            results = self.detector.model(frame, conf=0.5, verbose=False)
            result = results[0]
            frame = result.plot()

            # 更新检测信息
            if result.boxes is not None:
                num_objects = len(result.boxes)
                class_counts = {}
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = self.detector.class_names[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                info_text = f"检测到 {num_objects} 个对象: "
                for class_name, count in class_counts.items():
                    info_text += f"{class_name}: {count}个 "

                self.camera_info_label.setText(info_text)
            else:
                self.camera_info_label.setText("未检测到任何对象")

        # 如果正在录制，写入帧
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

        # 显示帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            QtCore.Qt.KeepAspectRatio
        )
        self.camera_label.setPixmap(scaled_pixmap)

    def toggle_camera_detection(self):
        self.is_camera_detecting = not self.is_camera_detecting
        if self.is_camera_detecting:
            self.detect_camera_btn.setText('⏹️ 停止检测')
            self.save_camera_btn.setEnabled(True)
            self.statusBar().showMessage('实时检测已开启')
        else:
            self.detect_camera_btn.setText('🔍 开始检测')
            self.statusBar().showMessage('实时检测已关闭')

    def save_camera_result(self):
        if not self.camera_active:
            return

        if not self.is_recording:
            # 开始录制
            self.is_recording = True
            self.save_camera_btn.setText('⏹️ 停止录制')
            self.statusBar().showMessage('开始录制摄像头画面...')

            # 获取摄像头属性
            fps = int(self.camera_cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:  # 如果无法获取FPS，使用默认值
                fps = 30

            width = int(self.camera_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建输出目录
            output_dir = "camera_recordings"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"recording_{timestamp}.mp4")

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 保存输出路径
            self.camera_result_path = output_path

        else:
            # 停止录制
            self.is_recording = False
            self.save_camera_btn.setText('💾 保存录像')
            self.statusBar().showMessage('录制已停止')

            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            # 添加到数据库记录
            self.db_manager.add_detection_record(
                self.user_id,
                'camera',
                '实时摄像头',
                self.camera_result_path
            )

            # 刷新历史记录
            self.load_history()

            QtWidgets.QMessageBox.information(self, '成功', f'录像已保存: {os.path.basename(self.camera_result_path)}')

    def closeEvent(self, event):
        # 弹出确认退出对话框
        reply = QtWidgets.QMessageBox.question(
            self, '确认退出',
            '确定要退出系统吗？',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            # 清理资源
            if self.video_cap:
                self.video_cap.release()
            if self.camera_cap:
                self.camera_cap.release()
            if self.video_timer.isActive():
                self.video_timer.stop()
            if self.camera_timer.isActive():
                self.camera_timer.stop()
            event.accept()
        else:
            event.ignore()