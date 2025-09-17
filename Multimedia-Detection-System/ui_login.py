from PyQt5 import QtWidgets, QtCore, QtGui
from database import DatabaseManager
from config import Config


class LoginWindow(QtWidgets.QDialog):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.current_user = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f'{Config.WINDOW_TITLE} - 登录')
        self.setFixedSize(950, 950)  # 稍微增加窗口尺寸
        self.setStyleSheet(self.get_enhanced_style())

        # 设置窗口图标
        self.setWindowIcon(QtGui.QIcon('icon.png'))  # 如果有图标文件的话

        # 主布局
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(25)  # 增加间距
        main_layout.setContentsMargins(50, 50, 50, 40)  # 增加边距

        # 标题
        title_label = QtWidgets.QLabel(Config.WINDOW_TITLE)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setFamily("Microsoft YaHei")  # 使用更现代的字体
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("""
            color: #2c3e50;
            padding: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2c3e50);
            border-radius: 10px;
            margin-bottom: 20px;
        """)
        main_layout.addWidget(title_label)

        # 创建选项卡
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("""
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
        """)

        # 登录标签页
        self.login_tab = QtWidgets.QWidget()
        self.setup_login_tab()
        self.tabs.addTab(self.login_tab, "用户登录")

        # 注册标签页
        self.register_tab = QtWidgets.QWidget()
        self.setup_register_tab()
        self.tabs.addTab(self.register_tab, "新用户注册")

        # 找回密码标签页
        self.forgot_tab = QtWidgets.QWidget()
        self.setup_forgot_tab()
        self.tabs.addTab(self.forgot_tab, "找回密码")

        main_layout.addWidget(self.tabs)

        # 底部版权信息
        copyright_label = QtWidgets.QLabel("© 现代化 小目标检测系统. 版权所有")
        copyright_label.setAlignment(QtCore.Qt.AlignCenter)
        copyright_label.setStyleSheet("""
            color: #7f8c8d;
            margin-top: 20px;
            font-size: 12px;
        """)
        main_layout.addWidget(copyright_label)

        self.setLayout(main_layout)

    def get_enhanced_style(self):
        """返回增强的样式表"""
        return """
            QMainWindow, QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ecf0f1, stop:1 #bdc3c7);
            }
            QLabel {
                color: #2c3e50;
                font-family: Microsoft YaHei;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                selection-background-color: #3498db;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: #eaf2f8;
            }
            QLineEdit:hover {
                border: 2px solid #3498db;
            }
            QComboBox {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
                min-height: 30px;
            }
            QComboBox:hover {
                border: 2px solid #3498db;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left-width: 1px;
                border-left-color: #bdc3c7;
                border-left-style: solid;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                border: 2px solid #3498db;
                selection-background-color: #3498db;
                background-color: white;
            }
        """

    def setup_login_tab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # 用户名输入
        username_label = QtWidgets.QLabel('用户名:')
        self.login_username = QtWidgets.QLineEdit()
        self.login_username.setPlaceholderText('请输入您的用户名')
        layout.addWidget(username_label)
        layout.addWidget(self.login_username)

        # 密码输入
        password_label = QtWidgets.QLabel('密码:')
        self.login_password = QtWidgets.QLineEdit()
        self.login_password.setPlaceholderText('请输入您的密码')
        self.login_password.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(password_label)
        layout.addWidget(self.login_password)

        # 登录按钮
        self.login_btn = QtWidgets.QPushButton('登 录')
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setDefault(True)
        self.login_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.login_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5dade2, stop:1 #3498db);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2c3e50, stop:1 #34495e);
                padding: 11px;
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        layout.addWidget(self.login_btn)

        # 状态标签
        self.login_status = QtWidgets.QLabel('')
        self.login_status.setAlignment(QtCore.Qt.AlignCenter)
        self.login_status.setStyleSheet("""
            color: #e74c3c;
            background-color: #fadbd8;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #e74c3c;
        """)
        layout.addWidget(self.login_status)

        self.login_tab.setLayout(layout)

    def setup_register_tab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        # 用户名输入
        username_label = QtWidgets.QLabel('用户名:')
        self.register_username = QtWidgets.QLineEdit()
        self.register_username.setPlaceholderText('请输入用户名 (至少2个字符)')
        layout.addWidget(username_label)
        layout.addWidget(self.register_username)

        # 密码输入
        password_label = QtWidgets.QLabel('密码:')
        self.register_password = QtWidgets.QLineEdit()
        self.register_password.setPlaceholderText('请输入密码 (至少6个字符)')
        self.register_password.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(password_label)
        layout.addWidget(self.register_password)

        # 确认密码
        confirm_label = QtWidgets.QLabel('确认密码:')
        self.register_confirm = QtWidgets.QLineEdit()
        self.register_confirm.setPlaceholderText('请再次输入密码')
        self.register_confirm.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(confirm_label)
        layout.addWidget(self.register_confirm)

        # 安全问题
        question_label = QtWidgets.QLabel('安全问题:')
        self.register_question = QtWidgets.QComboBox()
        self.register_question.addItems([
            '你最喜欢的颜色是什么？',
            '你的出生城市是哪里？',
            '你的第一所学校的名字是什么？',
            '你的宠物的名字是什么？',
            '你母亲的名字是什么？'
        ])
        layout.addWidget(question_label)
        layout.addWidget(self.register_question)

        # 安全问题答案
        answer_label = QtWidgets.QLabel('答案:')
        self.register_answer = QtWidgets.QLineEdit()
        self.register_answer.setPlaceholderText('请输入安全问题答案')
        layout.addWidget(answer_label)
        layout.addWidget(self.register_answer)

        # 注册按钮
        self.register_btn = QtWidgets.QPushButton('注 册')
        self.register_btn.clicked.connect(self.register)
        self.register_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.register_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ecc71, stop:1 #27ae60);
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #58d68d, stop:1 #2ecc71);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #239b56, stop:1 #1e8449);
                padding: 11px;
            }
        """)
        layout.addWidget(self.register_btn)

        # 状态标签
        self.register_status = QtWidgets.QLabel('')
        self.register_status.setAlignment(QtCore.Qt.AlignCenter)
        self.register_status.setStyleSheet("""
            color: #e74c3c;
            background-color: #fadbd8;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #e74c3c;
        """)
        layout.addWidget(self.register_status)

        self.register_tab.setLayout(layout)

    def setup_forgot_tab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        # 用户名输入
        username_label = QtWidgets.QLabel('用户名:')
        self.forgot_username = QtWidgets.QLineEdit()
        self.forgot_username.setPlaceholderText('请输入用户名')
        layout.addWidget(username_label)
        layout.addWidget(self.forgot_username)

        # 获取安全问题按钮
        self.get_question_btn = QtWidgets.QPushButton('获取安全问题')
        self.get_question_btn.clicked.connect(self.get_security_question)
        self.get_question_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.get_question_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f39c12, stop:1 #e67e22);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f4d03f, stop:1 #f39c12);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d35400, stop:1 #e67e22);
                padding: 9px;
            }
        """)
        layout.addWidget(self.get_question_btn)

        # 安全问题显示
        self.forgot_question = QtWidgets.QLabel('')
        self.forgot_question.setAlignment(QtCore.Qt.AlignCenter)
        self.forgot_question.setStyleSheet("""
            color: #16a085;
            background-color: #d1f2eb;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #16a085;
            font-weight: bold;
        """)
        layout.addWidget(self.forgot_question)

        # 安全问题答案
        answer_label = QtWidgets.QLabel('答案:')
        self.forgot_answer = QtWidgets.QLineEdit()
        self.forgot_answer.setPlaceholderText('请输入安全问题答案')
        layout.addWidget(answer_label)
        layout.addWidget(self.forgot_answer)

        # 新密码
        new_password_label = QtWidgets.QLabel('新密码:')
        self.forgot_new_password = QtWidgets.QLineEdit()
        self.forgot_new_password.setPlaceholderText('请输入新密码 (至少6个字符)')
        self.forgot_new_password.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(new_password_label)
        layout.addWidget(self.forgot_new_password)

        # 确认新密码
        confirm_label = QtWidgets.QLabel('确认密码:')
        self.forgot_confirm = QtWidgets.QLineEdit()
        self.forgot_confirm.setPlaceholderText('请再次输入新密码')
        self.forgot_confirm.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(confirm_label)
        layout.addWidget(self.forgot_confirm)

        # 重置密码按钮
        self.reset_btn = QtWidgets.QPushButton('重置密码')
        self.reset_btn.clicked.connect(self.reset_password)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9b59b6, stop:1 #8e44ad);
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #bb8fce, stop:1 #9b59b6);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6c3483, stop:1 #8e44ad);
                padding: 11px;
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        layout.addWidget(self.reset_btn)

        # 状态标签
        self.forgot_status = QtWidgets.QLabel('')
        self.forgot_status.setAlignment(QtCore.Qt.AlignCenter)
        self.forgot_status.setStyleSheet("""
            color: #e74c3c;
            background-color: #fadbd8;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #e74c3c;
        """)
        layout.addWidget(self.forgot_status)

        self.forgot_tab.setLayout(layout)

    def login(self):
        username = self.login_username.text().strip()
        password = self.login_password.text().strip()

        if not username or not password:
            self.login_status.setText('用户名和密码不能为空！')
            return

        user = self.db_manager.verify_user(username, password)
        if user:
            self.current_user = user
            self.login_status.setText('登录成功!')
            QtCore.QTimer.singleShot(500, self.accept_login)
        else:
            self.login_status.setText('用户名或密码错误！')

    def register(self):
        username = self.register_username.text().strip()
        password = self.register_password.text().strip()
        confirm = self.register_confirm.text().strip()
        question = self.register_question.currentText()
        answer = self.register_answer.text().strip()

        if not username or not password:
            self.register_status.setText('用户名和密码不能为空！')
            return

        if len(username) < 2:
            self.register_status.setText('用户名至少2个字符！')
            return

        if len(password) < 6:
            self.register_status.setText('密码至少6个字符！')
            return

        if password != confirm:
            self.register_status.setText('两次输入的密码不一致！')
            return

        if not answer:
            self.register_status.setText('安全问题答案不能为空！')
            return

        if self.db_manager.register_user(username, password, question, answer):
            self.register_status.setText('注册成功，请登录！')
            # 清空表单
            self.register_username.clear()
            self.register_password.clear()
            self.register_confirm.clear()
            self.register_answer.clear()
        else:
            self.register_status.setText('用户名已存在！')

    def get_security_question(self):
        username = self.forgot_username.text().strip()

        if not username:
            self.forgot_status.setText('请输入用户名')
            return

        question = self.db_manager.get_security_question(username)
        if question:
            self.forgot_question.setText(question)
            self.reset_btn.setEnabled(True)
            self.forgot_status.setText('')
        else:
            self.forgot_question.setText('')
            self.reset_btn.setEnabled(False)
            self.forgot_status.setText('用户名不存在！')

    def reset_password(self):
        username = self.forgot_username.text().strip()
        answer = self.forgot_answer.text().strip()
        new_password = self.forgot_new_password.text().strip()
        confirm = self.forgot_confirm.text().strip()

        if not answer:
            self.forgot_status.setText('请输入安全问题答案')
            return

        if not new_password:
            self.forgot_status.setText('请输入新密码')
            return

        if len(new_password) < 6:
            self.forgot_status.setText('密码至少6个字符')
            return

        if new_password != confirm:
            self.forgot_status.setText('两次输入的密码不一致')
            return

        if not self.db_manager.verify_security_answer(username, answer):
            self.forgot_status.setText('安全问题答案错误')
            return

        if self.db_manager.reset_password(username, new_password):
            self.forgot_status.setText('密码重置成功，请登录！')
            # 清空表单
            self.forgot_username.clear()
            self.forgot_answer.clear()
            self.forgot_new_password.clear()
            self.forgot_confirm.clear()
            self.forgot_question.setText('')
            self.reset_btn.setEnabled(False)
        else:
            self.forgot_status.setText('密码重置失败！')

    def accept_login(self):
        self.accept()