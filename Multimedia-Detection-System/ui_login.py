from PyQt5 import QtWidgets, QtCore, QtGui
import datetime
from database import register_user, verify_user, get_user_by_username, update_password, get_security_answers


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
        self.register_btn.clicked.connect(self.show_register_dialog)
        self.register_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.register_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)

        # 添加找回密码按钮
        self.forgot_btn = QtWidgets.QPushButton('找回密码')
        self.forgot_btn.clicked.connect(self.show_forgot_password_dialog)
        self.forgot_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.forgot_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)

        button_layout.addWidget(self.login_btn)
        button_layout.addWidget(self.register_btn)
        button_layout.addWidget(self.forgot_btn)
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
        self.setTabOrder(self.register_btn, self.forgot_btn)

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

    def show_register_dialog(self):
        """显示注册对话框"""
        dialog = RegisterDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            username, password, answers = dialog.get_data()

            if register_user(username, password, answers):
                self.status_label.setText('注册成功，请点击登录按钮！')
                self.animate_success()
            else:
                self.status_label.setText('该用户名已存在，请更换用户名！')
                self.animate_error()

    def show_forgot_password_dialog(self):
        """显示找回密码对话框"""
        dialog = ForgotPasswordDialog(self)
        dialog.exec_()

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


class RegisterDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('用户注册')
        self.setModal(True)
        self.setFixedSize(550, 600)  # 增加高度以适应新的布局
        self.init_ui()

    def init_ui(self):
        # 设置对话框背景
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(240, 245, 250))
        self.setPalette(palette)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QtWidgets.QLabel('用户注册')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 10px;
                background-color: rgba(255, 255, 255, 150);
                border-radius: 8px;
            }
        """)
        main_layout.addWidget(title_label)

        # 创建表单容器
        form_container = QtWidgets.QWidget()
        form_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 200);
                border-radius: 12px;
                border: 1px solid rgba(200, 200, 200, 100);
            }
        """)
        form_layout = QtWidgets.QVBoxLayout(form_container)
        form_layout.setSpacing(15)
        form_layout.setContentsMargins(25, 25, 25, 25)

        # 用户名输入
        username_layout = QtWidgets.QHBoxLayout()
        username_label = QtWidgets.QLabel('用户名:')
        username_label.setFixedWidth(80)
        username_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText('请输入用户名（至少2个字符）')
        self.username_input.setMinimumHeight(40)
        self.username_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        form_layout.addLayout(username_layout)

        # 密码输入
        password_layout = QtWidgets.QHBoxLayout()
        password_label = QtWidgets.QLabel('密码:')
        password_label.setFixedWidth(80)
        password_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setPlaceholderText('请输入密码（至少6个字符）')
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_input.setMinimumHeight(40)
        self.password_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        form_layout.addLayout(password_layout)

        # 确认密码
        confirm_password_layout = QtWidgets.QHBoxLayout()
        confirm_password_label = QtWidgets.QLabel('确认密码:')
        confirm_password_label.setFixedWidth(80)
        confirm_password_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        self.confirm_password_input = QtWidgets.QLineEdit()
        self.confirm_password_input.setPlaceholderText('请再次输入密码')
        self.confirm_password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirm_password_input.setMinimumHeight(40)
        self.confirm_password_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        confirm_password_layout.addWidget(confirm_password_label)
        confirm_password_layout.addWidget(self.confirm_password_input)
        form_layout.addLayout(confirm_password_layout)

        # 安全问题标题
        security_label = QtWidgets.QLabel('请设置安全问题（用于找回密码）:')
        security_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #34495e; background-color: transparent; padding: 5px;")
        form_layout.addWidget(security_label)

        # 安全问题1
        security1_layout = QtWidgets.QVBoxLayout()
        security1_question = QtWidgets.QLabel('1. 你最喜欢的颜色是什么？')
        security1_question.setStyleSheet("color: #34495e;")
        self.security1_answer = QtWidgets.QLineEdit()
        self.security1_answer.setPlaceholderText('请输入答案')
        self.security1_answer.setMinimumHeight(35)
        self.security1_answer.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        security1_layout.addWidget(security1_question)
        security1_layout.addWidget(self.security1_answer)
        form_layout.addLayout(security1_layout)

        # 安全问题2
        security2_layout = QtWidgets.QVBoxLayout()
        security2_question = QtWidgets.QLabel('2. 你的爱好是什么？')
        security2_question.setStyleSheet("color: #34495e;")
        self.security2_answer = QtWidgets.QLineEdit()
        self.security2_answer.setPlaceholderText('请输入答案')
        self.security2_answer.setMinimumHeight(35)
        self.security2_answer.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        security2_layout.addWidget(security2_question)
        security2_layout.addWidget(self.security2_answer)
        form_layout.addLayout(security2_layout)

        # 安全问题3
        security3_layout = QtWidgets.QVBoxLayout()
        security3_question = QtWidgets.QLabel('3. 你的生日是哪天？(格式:YYYY-MM-DD)')
        security3_question.setStyleSheet("color: #34495e;")
        self.security3_answer = QtWidgets.QLineEdit()
        self.security3_answer.setPlaceholderText('请输入答案，例如:1990-01-01')
        self.security3_answer.setMinimumHeight(35)
        self.security3_answer.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        security3_layout.addWidget(security3_question)
        security3_layout.addWidget(self.security3_answer)
        form_layout.addLayout(security3_layout)

        main_layout.addWidget(form_container)

        # 按钮布局
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(20)

        self.register_btn = QtWidgets.QPushButton('注册')
        self.register_btn.clicked.connect(self.accept)
        self.register_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.register_btn.setMinimumHeight(40)
        self.register_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)

        self.cancel_btn = QtWidgets.QPushButton('取消')
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)

        button_layout.addWidget(self.register_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)

        # 状态标签
        self.status_label = QtWidgets.QLabel('')
        self.status_label.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 150);
                padding: 10px;
                border-radius: 6px;
            }
        """)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def get_data(self):
        return (
            self.username_input.text().strip(),
            self.password_input.text().strip(),
            (
                self.security1_answer.text().strip(),
                self.security2_answer.text().strip(),
                self.security3_answer.text().strip()
            )
        )

    def accept(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        confirm_password = self.confirm_password_input.text().strip()
        answer1 = self.security1_answer.text().strip()
        answer2 = self.security2_answer.text().strip()
        answer3 = self.security3_answer.text().strip()

        # 验证输入
        if not username or not password or not confirm_password:
            self.status_label.setText('用户名和密码不能为空！')
            return

        if len(username) < 2:
            self.status_label.setText('用户名至少使用2个字符！')
            return

        if len(password) < 6:
            self.status_label.setText('密码至少使用6个字符！')
            return

        if password != confirm_password:
            self.status_label.setText('两次输入的密码不一致！')
            return

        if not answer1 or not answer2 or not answer3:
            self.status_label.setText('所有安全问题都必须填写！')
            return

        # 验证生日格式
        if not self.validate_birthday(answer3):
            self.status_label.setText('生日格式不正确，请使用YYYY-MM-DD格式！')
            return

        super().accept()

    def validate_birthday(self, birthday):
        try:
            datetime.datetime.strptime(birthday, '%Y-%m-%d')
            return True
        except ValueError:
            return False


class ForgotPasswordDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('找回密码')
        self.setModal(True)
        self.setFixedSize(550, 600)  # 增加高度以适应新的布局
        self.username = None
        self.init_ui()

    def init_ui(self):
        # 设置对话框背景
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(240, 245, 250))
        self.setPalette(palette)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QtWidgets.QLabel('找回密码')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 10px;
                background-color: rgba(255, 255, 255, 150);
                border-radius: 8px;
            }
        """)
        self.layout.addWidget(title_label)

        # 用户名输入
        username_layout = QtWidgets.QHBoxLayout()
        username_label = QtWidgets.QLabel('用户名:')
        username_label.setFixedWidth(80)
        username_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText('请输入您的用户名')
        self.username_input.setMinimumHeight(40)
        self.username_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        self.username_input.textChanged.connect(self.on_username_changed)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        self.layout.addLayout(username_layout)

        # 安全问题容器（初始隐藏）
        self.security_container = QtWidgets.QWidget()
        self.security_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 200);
                border-radius: 12px;
                border: 1px solid rgba(200, 200, 200, 100);
            }
        """)
        self.security_layout = QtWidgets.QVBoxLayout(self.security_container)
        self.security_layout.setSpacing(15)
        self.security_layout.setContentsMargins(25, 25, 25, 25)

        # 安全问题1
        security1_layout = QtWidgets.QVBoxLayout()
        security1_question = QtWidgets.QLabel('1. 你最喜欢的颜色是什么？')
        security1_question.setStyleSheet("color: #34495e;")
        self.security1_answer = QtWidgets.QLineEdit()
        self.security1_answer.setPlaceholderText('请输入答案')
        self.security1_answer.setMinimumHeight(35)
        self.security1_answer.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        security1_layout.addWidget(security1_question)
        security1_layout.addWidget(self.security1_answer)
        self.security_layout.addLayout(security1_layout)

        # 安全问题2
        security2_layout = QtWidgets.QVBoxLayout()
        security2_question = QtWidgets.QLabel('2. 你的爱好是什么？')
        security2_question.setStyleSheet("color: #34495e;")
        self.security2_answer = QtWidgets.QLineEdit()
        self.security2_answer.setPlaceholderText('请输入答案')
        self.security2_answer.setMinimumHeight(35)
        self.security2_answer.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        security2_layout.addWidget(security2_question)
        security2_layout.addWidget(self.security2_answer)
        self.security_layout.addLayout(security2_layout)

        # 安全问题3
        security3_layout = QtWidgets.QVBoxLayout()
        security3_question = QtWidgets.QLabel('3. 你的生日是哪天？(格式:YYYY-MM-DD)')
        security3_question.setStyleSheet("color: #34495e;")
        self.security3_answer = QtWidgets.QLineEdit()
        self.security3_answer.setPlaceholderText('请输入答案，例如:1990-01-01')
        self.security3_answer.setMinimumHeight(35)
        self.security3_answer.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        security3_layout.addWidget(security3_question)
        security3_layout.addWidget(self.security3_answer)
        self.security_layout.addLayout(security3_layout)

        # 新密码
        new_password_layout = QtWidgets.QHBoxLayout()
        new_password_label = QtWidgets.QLabel('新密码:')
        new_password_label.setFixedWidth(80)
        new_password_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        self.new_password_input = QtWidgets.QLineEdit()
        self.new_password_input.setPlaceholderText('请输入新密码')
        self.new_password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.new_password_input.setMinimumHeight(35)
        self.new_password_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        new_password_layout.addWidget(new_password_label)
        new_password_layout.addWidget(self.new_password_input)
        self.security_layout.addLayout(new_password_layout)

        # 确认新密码
        confirm_password_layout = QtWidgets.QHBoxLayout()
        confirm_password_label = QtWidgets.QLabel('确认密码:')
        confirm_password_label.setFixedWidth(80)
        confirm_password_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        self.confirm_password_input = QtWidgets.QLineEdit()
        self.confirm_password_input.setPlaceholderText('请再次输入新密码')
        self.confirm_password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirm_password_input.setMinimumHeight(35)
        self.confirm_password_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dcdcdc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        confirm_password_layout.addWidget(confirm_password_label)
        confirm_password_layout.addWidget(self.confirm_password_input)
        self.security_layout.addLayout(confirm_password_layout)

        # 重置按钮
        self.reset_btn = QtWidgets.QPushButton('重置密码')
        self.reset_btn.clicked.connect(self.reset_password)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.security_layout.addWidget(self.reset_btn)

        self.security_container.setVisible(False)
        self.layout.addWidget(self.security_container)

        # 按钮布局
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(20)

        self.verify_btn = QtWidgets.QPushButton('验证用户')
        self.verify_btn.clicked.connect(self.verify_user)
        self.verify_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.verify_btn.setMinimumHeight(40)
        self.verify_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)

        self.cancel_btn = QtWidgets.QPushButton('取消')
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)

        button_layout.addWidget(self.verify_btn)
        button_layout.addWidget(self.cancel_btn)
        self.layout.addLayout(button_layout)

        # 状态标签
        self.status_label = QtWidgets.QLabel('')
        self.status_label.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                font-weight: bold;
                background-color: rgba(255, 255, 255, 150);
                padding: 10px;
                border-radius: 6px;
            }
        """)
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)

    def on_username_changed(self):
        """当用户名改变时，隐藏安全问题区域"""
        self.security_container.setVisible(False)
        self.reset_btn.setEnabled(False)

    def verify_user(self):
        """验证用户是否存在"""
        username = self.username_input.text().strip()

        if not username:
            self.status_label.setText('请输入用户名!')
            return

        # 检查用户是否存在
        user = get_user_by_username(username)
        if not user:
            self.status_label.setText('用户名不存在!')
            return

        self.username = username
        self.status_label.setText('用户验证成功，请回答安全问题!')
        self.security_container.setVisible(True)
        self.reset_btn.setEnabled(True)

    def reset_password(self):
        """重置密码"""
        answer1 = self.security1_answer.text().strip()
        answer2 = self.security2_answer.text().strip()
        answer3 = self.security3_answer.text().strip()
        new_password = self.new_password_input.text().strip()
        confirm_password = self.confirm_password_input.text().strip()

        # 验证输入
        if not answer1 or not answer2 or not answer3:
            self.status_label.setText('所有安全问题都必须回答!')
            return

        if not new_password or not confirm_password:
            self.status_label.setText('新密码不能为空!')
            return

        if new_password != confirm_password:
            self.status_label.setText('两次输入的密码不一致!')
            return

        if len(new_password) < 6:
            self.status_label.setText('密码长度至少为6个字符!')
            return

        # 获取存储的安全问题答案
        stored_answers = get_security_answers(self.username)
        if not stored_answers:
            self.status_label.setText('获取安全问题答案失败!')
            return

        # 验证安全问题答案
        if (answer1 != stored_answers[0] or
                answer2 != stored_answers[1] or
                answer3 != stored_answers[2]):
            self.status_label.setText('安全问题答案不正确!')
            return

        # 更新密码
        if update_password(self.username, new_password):
            self.status_label.setText('密码重置成功，请使用新密码登录!')
            QtCore.QTimer.singleShot(2000, self.accept)
        else:
            self.status_label.setText('密码重置失败，请稍后重试!')
