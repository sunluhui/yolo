from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import os
from datetime import datetime
from threading import Thread
from config import Config


class MainWindow(QtWidgets.QMainWindow):
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
        self.is_recording = False
        self.video_writer = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f'{Config.WINDOW_TITLE} - 欢迎 {self.username}')
        self.setGeometry(100, 100, *Config.WINDOW_SIZE)
        self.setStyleSheet(Config.THEME_STYLE)

        # 中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 选项卡
        self.tabs = QtWidgets.QTabWidget()

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

        # 历史记录标签页
        self.history_tab = QtWidgets.QWidget()
        self.setup_history_tab()
        self.tabs.addTab(self.history_tab, "历史记录")

        main_layout.addWidget(self.tabs)

        # 状态栏
        self.statusBar().showMessage('就绪')

    def setup_image_tab(self):
        layout = QtWidgets.QVBoxLayout(self.image_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 控制按钮
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(10)

        self.select_image_btn = QtWidgets.QPushButton('选择图片')
        self.select_image_btn.clicked.connect(self.select_image)
        control_layout.addWidget(self.select_image_btn)

        self.detect_image_btn = QtWidgets.QPushButton('开始检测')
        self.detect_image_btn.clicked.connect(self.detect_image)
        self.detect_image_btn.setEnabled(False)
        control_layout.addWidget(self.detect_image_btn)

        self.save_image_btn = QtWidgets.QPushButton('保存结果')
        self.save_image_btn.clicked.connect(self.save_image_result)
        self.save_image_btn.setEnabled(False)
        control_layout.addWidget(self.save_image_btn)

        layout.addLayout(control_layout)

        # 图片显示区域
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.setSpacing(20)

        # 原始图片
        original_frame = QtWidgets.QGroupBox("原始图像")
        original_layout = QtWidgets.QVBoxLayout()
        self.original_image_label = QtWidgets.QLabel()
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_image_label.setMinimumSize(640, 480)
        self.original_image_label.setStyleSheet('border: 1px solid #cccccc; background-color: #f8f8f8;')
        self.original_image_label.setText('请选择图片')
        original_layout.addWidget(self.original_image_label)
        original_frame.setLayout(original_layout)
        image_layout.addWidget(original_frame)

        # 检测结果图片
        result_frame = QtWidgets.QGroupBox("检测结果")
        result_layout = QtWidgets.QVBoxLayout()
        self.result_image_label = QtWidgets.QLabel()
        self.result_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_image_label.setMinimumSize(640, 480)
        self.result_image_label.setStyleSheet('border: 1px solid #cccccc; background-color: #f8f8f8;')
        self.result_image_label.setText('检测结果将显示在这里')
        result_layout.addWidget(self.result_image_label)
        result_frame.setLayout(result_layout)
        image_layout.addWidget(result_frame)

        layout.addLayout(image_layout)

        # 检测信息
        info_frame = QtWidgets.QGroupBox("检测信息")
        info_layout = QtWidgets.QVBoxLayout()
        self.image_info_label = QtWidgets.QLabel('检测信息将显示在这里')
        self.image_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.image_info_label.setWordWrap(True)
        self.image_info_label.setStyleSheet('padding: 10px; background-color: #f8f8f8; border-radius: 4px;')
        info_layout.addWidget(self.image_info_label)
        info_frame.setLayout(info_layout)
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
        control_layout.setSpacing(10)

        self.select_video_btn = QtWidgets.QPushButton('选择视频')
        self.select_video_btn.clicked.connect(self.select_video)
        control_layout.addWidget(self.select_video_btn)

        self.play_video_btn = QtWidgets.QPushButton('播放/暂停')
        self.play_video_btn.clicked.connect(self.toggle_video_play)
        self.play_video_btn.setEnabled(False)
        control_layout.addWidget(self.play_video_btn)

        self.detect_video_btn = QtWidgets.QPushButton('开始检测')
        self.detect_video_btn.clicked.connect(self.detect_video)
        self.detect_video_btn.setEnabled(False)
        control_layout.addWidget(self.detect_video_btn)

        self.save_video_btn = QtWidgets.QPushButton('保存结果')
        self.save_video_btn.clicked.connect(self.save_video_result)
        self.save_video_btn.setEnabled(False)
        control_layout.addWidget(self.save_video_btn)

        layout.addLayout(control_layout)

        # 视频显示区域
        video_layout = QtWidgets.QHBoxLayout()
        video_layout.setSpacing(20)

        # 原始视频
        original_frame = QtWidgets.QGroupBox("原始视频")
        original_layout = QtWidgets.QVBoxLayout()
        self.original_video_label = QtWidgets.QLabel()
        self.original_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_video_label.setMinimumSize(640, 480)
        self.original_video_label.setStyleSheet('border: 1px solid #cccccc; background-color: #f8f8f8;')
        self.original_video_label.setText('请选择视频')
        original_layout.addWidget(self.original_video_label)
        original_frame.setLayout(original_layout)
        video_layout.addWidget(original_frame)

        # 检测结果视频
        result_frame = QtWidgets.QGroupBox("检测结果")
        result_layout = QtWidgets.QVBoxLayout()
        self.result_video_label = QtWidgets.QLabel()
        self.result_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_video_label.setMinimumSize(640, 480)
        self.result_video_label.setStyleSheet('border: 1px solid #cccccc; background-color: #f8f8f8;')
        self.result_video_label.setText('检测结果将显示在这里')
        result_layout.addWidget(self.result_video_label)
        result_frame.setLayout(result_layout)
        video_layout.addWidget(result_frame)

        layout.addLayout(video_layout)

        # 检测信息
        info_frame = QtWidgets.QGroupBox("检测信息")
        info_layout = QtWidgets.QVBoxLayout()
        self.video_info_label = QtWidgets.QLabel('检测信息将显示在这里')
        self.video_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet('padding: 10px; background-color: #f8f8f8; border-radius: 4px;')
        info_layout.addWidget(self.video_info_label)
        info_frame.setLayout(info_layout)
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
        control_layout.setSpacing(10)

        self.start_camera_btn = QtWidgets.QPushButton('开启摄像头')
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.start_camera_btn)

        self.detect_camera_btn = QtWidgets.QPushButton('开始检测')
        self.detect_camera_btn.clicked.connect(self.toggle_camera_detection)
        self.detect_camera_btn.setEnabled(False)
        control_layout.addWidget(self.detect_camera_btn)

        self.save_camera_btn = QtWidgets.QPushButton('开始录制')
        self.save_camera_btn.clicked.connect(self.toggle_camera_recording)
        self.save_camera_btn.setEnabled(False)
        control_layout.addWidget(self.save_camera_btn)

        layout.addLayout(control_layout)

        # 摄像头显示区域
        camera_frame = QtWidgets.QGroupBox("摄像头画面")
        camera_layout = QtWidgets.QVBoxLayout()
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet('border: 1px solid #cccccc; background-color: #f8f8f8;')
        self.camera_label.setText('摄像头未开启')
        camera_layout.addWidget(self.camera_label)
        camera_frame.setLayout(camera_layout)
        layout.addWidget(camera_frame)

        # 检测信息
        info_frame = QtWidgets.QGroupBox("检测信息")
        info_layout = QtWidgets.QVBoxLayout()
        self.camera_info_label = QtWidgets.QLabel('摄像头就绪')
        self.camera_info_label.setAlignment(QtCore.Qt.AlignLeft)
        self.camera_info_label.setWordWrap(True)
        self.camera_info_label.setStyleSheet('padding: 10px; background-color: #f8f8f8; border-radius: 4px;')
        info_layout.addWidget(self.camera_info_label)
        info_frame.setLayout(info_layout)
        layout.addWidget(info_frame)

        self.camera_tab.setLayout(layout)

        self.camera_result_path = None
        self.camera_detection_data = None

        # 摄像头定时器
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.camera_cap = None
        self.is_camera_detecting = False

    def setup_history_tab(self):
        layout = QtWidgets.QVBoxLayout(self.history_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QtWidgets.QLabel('检测历史记录')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # 历史记录表格
        self.history_table = QtWidgets.QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(['检测类型', '源文件', '结果文件', '检测时间'])
        self.history_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.history_table.setAlternatingRowColors(True)
        layout.addWidget(self.history_table)

        # 刷新按钮
        self.refresh_btn = QtWidgets.QPushButton('刷新记录')
        self.refresh_btn.clicked.connect(self.load_history)
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
            self.history_table.setItem(row, 3, QtWidgets.QTableWidgetItem(detection_time))

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
            self.play_video_btn.setText('暂停')
        else:
            self.video_timer.stop()
            self.is_playing = False
            self.play_video_btn.setText('播放')

    def update_video_frame(self):
        if self.video_cap is None:
            return

        ret, frame = self.video_cap.read()
        if not ret:
            self.video_timer.stop()
            self.is_playing = False
            self.play_video_btn.setText('播放')
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
        self.setEnabled(False)  # 禁用界面防止重复操作
        QtWidgets.QApplication.processEvents()  # 更新UI

        # 在子线程中执行检测
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
                    QtCore.Qt.KeepAspectRatio
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
            self.db_manager.add_detection_record(
                self.user_id,
                'video',
                self.video_path,
                self.video_result_path
            )

            # 刷新历史记录
            self.load_history()

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
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, '错误', f'保存失败: {str(e)}')

    def toggle_camera(self):
        if not self.camera_active:
            # 开启摄像头
            self.camera_cap = cv2.VideoCapture(0)
            if not self.camera_cap.isOpened():
                QtWidgets.QMessageBox.warning(self, '错误', '无法打开摄像头')
                return

            # 获取摄像头参数
            self.camera_fps = int(self.camera_cap.get(cv2.CAP_PROP_FPS))
            if self.camera_fps <= 0:
                self.camera_fps = 30  # 默认帧率

            self.camera_width = int(self.camera_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.camera_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.camera_timer.start(1000 // self.camera_fps)  # 根据帧率设置定时器
            self.camera_active = True
            self.start_camera_btn.setText('关闭摄像头')
            self.detect_camera_btn.setEnabled(True)
            self.save_camera_btn.setEnabled(True)
            self.statusBar().showMessage('摄像头已开启')
        else:
            # 停止录制（如果正在录制）
            if self.is_recording:
                self.toggle_camera_recording()

            # 关闭摄像头
            self.camera_timer.stop()
            if self.camera_cap:
                self.camera_cap.release()
                self.camera_cap = None
            self.camera_active = False
            self.start_camera_btn.setText('开启摄像头')
            self.detect_camera_btn.setEnabled(False)
            self.detect_camera_btn.setText('开始检测')
            self.save_camera_btn.setEnabled(False)
            self.save_camera_btn.setText('开始录制')
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

        # 保存原始帧用于录制
        original_frame = frame.copy()

        if self.is_camera_detecting:
            # 进行实时检测
            results = self.detector.model(frame, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
            result = results[0]
            frame = result.plot()

        # 如果正在录制，写入帧到视频文件
        if self.is_recording and self.video_writer:
            self.video_writer.write(original_frame)

            # 显示录制状态
            recording_time = (datetime.now() - self.record_start_time).total_seconds()
            self.camera_info_label.setText(f'录制中: {recording_time:.1f}秒')

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
            self.detect_camera_btn.setText('停止检测')
            self.statusBar().showMessage('实时检测已开启')
        else:
            self.detect_camera_btn.setText('开始检测')
            self.statusBar().showMessage('实时检测已关闭')

    def toggle_camera_recording(self):
        if not self.is_recording:
            # 开始录制
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, '保存摄像头录像',
                f"camera_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                'MP4视频 (*.mp4);;AVI视频 (*.avi)'
            )

            if save_path:
                # 创建视频编码器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    save_path, fourcc, self.camera_fps,
                    (self.camera_width, self.camera_height)
                )
                self.is_recording = True
                self.record_start_time = datetime.now()
                self.save_camera_btn.setText('停止录制')
                self.statusBar().showMessage('正在录制...')
        else:
            # 停止录制
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.is_recording = False
            self.save_camera_btn.setText('开始录制')
            recording_duration = (datetime.now() - self.record_start_time).total_seconds()
            self.statusBar().showMessage(f'录制完成，时长: {recording_duration:.1f}秒')

            # 添加到数据库记录
            save_path = self.video_writer.__dict__.get('filename', '未知路径')
            self.db_manager.add_detection_record(
                self.user_id,
                'camera',
                '实时摄像头',
                save_path
            )

            # 刷新历史记录
            self.load_history()

    def closeEvent(self, event):
        # 退出确认
        reply = QtWidgets.QMessageBox.question(
            self, '确认退出',
            '确定要退出小目标检测系统吗？',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            # 清理资源
            if self.video_cap:
                self.video_cap.release()
            if self.camera_cap:
                self.camera_cap.release()
            if self.video_writer:
                self.video_writer.release()
            if self.video_timer.isActive():
                self.video_timer.stop()
            if self.camera_timer.isActive():
                self.camera_timer.stop()
            event.accept()
        else:
            event.ignore()