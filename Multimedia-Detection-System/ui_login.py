from PyQt5 import QtWidgets, QtCore, QtGui
from database import register_user, verify_user


class LoginWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.current_user = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('小目标检测系统')
        self.setGeometry(400, 400, 500, 500)  # 稍微增大窗口尺寸

        # 设置应用程序样式
        self.setStyleSheet("""
            QWidget {
                font-family: 'Microsoft YaHei', Arial, sans-serif;
            }
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                background-color: rgba(255, 255, 255, 200);
                selection-background-color: #4CAF50;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
                background-color: rgba(255, 255, 255, 230);
            }
            QPushButton {
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                min-width: 100px;
            }
            QPushButton:hover {
                opacity: 0.9;
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                transform: translateY(1px);
            }
            QLabel {
                color: #333333;
                font-weight: 500;
            }
        """)

        # 设置背景图片
        self.setAutoFillBackground(True)
        palette = self.palette()
        background_image = QtGui.QPixmap("/Users/dell/Desktop/beijing.jpg")
        # 缩放背景图片以适应窗口
        scaled_bg = background_image.scaled(self.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(scaled_bg))
        self.setPalette(palette)

        # 创建主布局
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # 标题
        title_label = QtWidgets.QLabel('小目标检测系统')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_font.setFamily('Microsoft YaHei')
        title_label.setFont(title_font)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                background-color: rgba(255, 255, 255, 180);
                padding: 15px;
                border-radius: 10px;
            }
        """)
        main_layout.addWidget(title_label)

        # 创建表单容器
        form_container = QtWidgets.QWidget()
        form_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 200);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 150);
            }
        """)
        form_layout = QtWidgets.QVBoxLayout(form_container)
        form_layout.setSpacing(20)
        form_layout.setContentsMargins(30, 30, 30, 30)

        # 用户名输入
        username_layout = QtWidgets.QHBoxLayout()
        username_label = QtWidgets.QLabel('用户名:')
        username_label.setFixedWidth(80)
        username_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText('请输入用户名')
        self.username_input.setMinimumHeight(40)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        form_layout.addLayout(username_layout)

        # 密码输入
        password_layout = QtWidgets.QHBoxLayout()
        password_label = QtWidgets.QLabel('密码:')
        password_label.setFixedWidth(80)
        password_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setPlaceholderText('请输入密码')
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_input.setMinimumHeight(40)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        form_layout.addLayout(password_layout)

        # 按钮布局
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(20)

        self.login_btn = QtWidgets.QPushButton('登录')
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setDefault(True)
        self.login_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.register_btn = QtWidgets.QPushButton('注册')
        self.register_btn.clicked.connect(self.register)
        self.register_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.register_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)

        button_layout.addWidget(self.login_btn)
        button_layout.addWidget(self.register_btn)
        form_layout.addLayout(button_layout)

        # 将表单容器添加到主布局
        main_layout.addWidget(form_container)

        # 状态标签
        self.status_label = QtWidgets.QLabel('')
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 180);
                padding: 10px;
                border-radius: 8px;
            }
        """)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

        # 设置Tab键顺序
        self.setTabOrder(self.username_input, self.password_input)
        self.setTabOrder(self.password_input, self.login_btn)
        self.setTabOrder(self.login_btn, self.register_btn)

        # 添加快捷键支持
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self, self.login)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self, self.login)

    def resizeEvent(self, event):
        # 重写resizeEvent方法，使背景图片随窗口大小变化而缩放
        palette = self.palette()
        background_image = QtGui.QPixmap("/Users/dell/Desktop/beijing.jpg")
        scaled_bg = background_image.scaled(self.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(scaled_bg))
        self.setPalette(palette)
        super().resizeEvent(event)

    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            self.status_label.setText('用户名和密码都不能为空！')
            return

        user = verify_user(username, password)
        if user:
            self.current_user = user
            self.status_label.setText('登录成功!')
            # 添加登录成功动画效果
            self.animate_success()
            QtCore.QTimer.singleShot(1000, self.accept_login)
        else:
            self.status_label.setText('用户名或密码错误！')
            # 添加错误提示动画
            self.animate_error()

    def register(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            self.status_label.setText('用户名和密码都不能为空！')
            self.animate_error()
            return

        if len(username) < 3:
            self.status_label.setText('用户名至少使用3个字符！')
            self.animate_error()
            return

        if len(password) < 6:
            self.status_label.setText('密码至少使用6个字符！')
            self.animate_error()
            return

        if register_user(username, password):
            self.status_label.setText('注册成功，请点击登录按钮！')
            self.animate_success()
        else:
            self.status_label.setText('该用户名已存在，请更换用户名！')
            self.animate_error()

    def animate_success(self):
        # 成功动画效果
        animation = QtCore.QPropertyAnimation(self.status_label, b"geometry")
        animation.setDuration(200)
        animation.setKeyValueAt(0, self.status_label.geometry())
        animation.setKeyValueAt(0.5, QtCore.QRect(
            self.status_label.x() - 5,
            self.status_label.y(),
            self.status_label.width(),
            self.status_label.height()
        ))
        animation.setKeyValueAt(1, self.status_label.geometry())
        animation.start()

        # 改变状态标签颜色为成功色
        self.status_label.setStyleSheet("""
            QLabel {
                color: #27ae60;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 180);
                padding: 10px;
                border-radius: 8px;
            }
        """)

    def animate_error(self):
        # 错误动画效果
        animation = QtCore.QPropertyAnimation(self.status_label, b"geometry")
        animation.setDuration(200)
        animation.setKeyValueAt(0, self.status_label.geometry())
        animation.setKeyValueAt(0.25, QtCore.QRect(
            self.status_label.x() - 5,
            self.status_label.y(),
            self.status_label.width(),
            self.status_label.height()
        ))
        animation.setKeyValueAt(0.5, QtCore.QRect(
            self.status_label.x() + 5,
            self.status_label.y(),
            self.status_label.width(),
            self.status_label.height()
        ))
        animation.setKeyValueAt(0.75, QtCore.QRect(
            self.status_label.x() - 5,
            self.status_label.y(),
            self.status_label.width(),
            self.status_label.height()
        ))
        animation.setKeyValueAt(1, self.status_label.geometry())
        animation.start()

        # 恢复状态标签颜色为错误色
        self.status_label.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 180);
                padding: 10px;
                border-radius: 8px;
            }
        """)

    def accept_login(self):
        self.close()