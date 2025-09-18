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

        # 标题 - 增大字体并美化
        title_label = QtWidgets.QLabel(Config.WINDOW_TITLE)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setFamily("Microsoft YaHei")
        title_font.setPointSize(32)  # 增大字体
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("""
            color: white;
            padding: 20px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2c3e50);
            border-radius: 15px;
            margin-bottom: 30px;
            border: 2px solid #2980b9;
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
        self.tabs.addTab(self.login_tab, "🔐用户登录")

        # 注册标签页
        self.register_tab = QtWidgets.QWidget()
        self.setup_register_tab()
        self.tabs.addTab(self.register_tab, "👤新用户注册")

        # 找回密码标签页
        self.forgot_tab = QtWidgets.QWidget()
        self.setup_forgot_tab()
        self.tabs.addTab(self.forgot_tab, "🔓找回密码")

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
        layout.setSpacing(25)  # 增加间距
        layout.setContentsMargins(40, 40, 40, 30)  # 调整边距

        # 用户名输入 - 优化设计
        username_container = QtWidgets.QWidget()
        username_container.setStyleSheet("background: transparent;")
        username_layout = QtWidgets.QVBoxLayout(username_container)
        username_layout.setContentsMargins(0, 0, 0, 0)
        username_layout.setSpacing(8)

        username_label = QtWidgets.QLabel('用户名：')
        username_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-weight: bold;
                font-size: 14px;
                padding-left: 5px;
            }
        """)

        self.login_username = QtWidgets.QLineEdit()
        self.login_username.setPlaceholderText('请输入您的用户名')
        self.login_username.setMinimumHeight(45)
        self.login_username.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                padding: 12px 15px;
                font-size: 14px;
                color: #2c3e50;
                selection-background-color: #3498db;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: #f8fafd;
            }
            QLineEdit:hover {
                border: 2px solid #a0aec0;
            }
        """)

        username_layout.addWidget(username_label)
        username_layout.addWidget(self.login_username)
        layout.addWidget(username_container)

        # 密码输入 - 优化设计
        password_container = QtWidgets.QWidget()
        password_container.setStyleSheet("background: transparent;")
        password_layout = QtWidgets.QVBoxLayout(password_container)
        password_layout.setContentsMargins(0, 0, 0, 0)
        password_layout.setSpacing(8)

        password_label = QtWidgets.QLabel('密码：')
        password_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-weight: bold;
                font-size: 14px;
                padding-left: 5px;
            }
        """)

        self.login_password = QtWidgets.QLineEdit()
        self.login_password.setPlaceholderText('请输入您的密码')
        self.login_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.login_password.setMinimumHeight(45)
        self.login_password.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                padding: 12px 15px;
                font-size: 14px;
                color: #2c3e50;
                selection-background-color: #3498db;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: #f8fafd;
            }
            QLineEdit:hover {
                border: 2px solid #a0aec0;
            }
        """)

        # 添加显示/隐藏密码按钮
        toggle_password_btn = QtWidgets.QToolButton()
        toggle_password_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        toggle_password_btn.setText("👁")
        toggle_password_btn.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                font-size: 16px;
                padding: 0px;
                color: #718096;
            }
            QToolButton:hover {
                color: #3498db;
            }
        """)
        toggle_password_btn.clicked.connect(
            lambda: self.toggle_password_visibility(self.login_password, toggle_password_btn))

        # 创建带按钮的密码框
        password_with_button = QtWidgets.QHBoxLayout()
        password_with_button.addWidget(self.login_password)
        password_with_button.addWidget(toggle_password_btn)
        password_with_button.setContentsMargins(0, 0, 0, 0)

        password_layout.addWidget(password_label)
        password_layout.addLayout(password_with_button)
        layout.addWidget(password_container)

        # 登录按钮 - 保持原有样式，但调整上边距
        self.login_btn = QtWidgets.QPushButton('登 录')
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setDefault(True)
        self.login_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.login_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                    color: white;
                    border: none;
                    padding: 15px;
                    border-radius: 10px;
                    font-weight: bold;
                    font-size: 16px;
                    margin-top: 10px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5dade2, stop:1 #3498db);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2c3e50, stop:1 #34495e);
                    padding: 14px;
                }
                QPushButton:disabled {
                    background: #e2e8f0;
                    color: #a0aec0;
                }
            """)
        layout.addWidget(self.login_btn)

        # 状态标签 - 微调样式
        self.login_status = QtWidgets.QLabel('')
        self.login_status.setAlignment(QtCore.Qt.AlignCenter)
        self.login_status.setStyleSheet("""
                QLabel {
                    color: #e53e3e;
                    background-color: #fed7d7;
                    padding: 12px;
                    border-radius: 8px;
                    border: 1px solid #feb2b2;
                    font-size: 13px;
                    margin-top: 10px;
                }
            """)
        layout.addWidget(self.login_status)

        self.login_tab.setLayout(layout)

    def toggle_password_visibility(self, password_field, toggle_button):
        if password_field.echoMode() == QtWidgets.QLineEdit.Password:
            password_field.setEchoMode(QtWidgets.QLineEdit.Normal)
            toggle_button.setText("🔒")
        else:
            password_field.setEchoMode(QtWidgets.QLineEdit.Password)
            toggle_button.setText("👁")

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

        # 设置下拉菜单样式
        self.register_question.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                min-height: 40px;
                color: #2c3e50;
                font-family: Microsoft YaHei;
            }
            QComboBox:hover {
                border: 2px solid #3498db;
                background-color: #f8f9fa;
            }
            QComboBox:focus {
                border: 2px solid #3498db;
                background-color: #eaf2f8;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left-width: 1px;
                border-left-color: #bdc3c7;
                border-left-style: solid;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid #6c757d;
                width: 0;
                height: 0;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                border: 2px solid #3498db;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #3498db;
                selection-color: white;
                outline: none;
                padding: 5px;
                margin: 0px;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px 12px;
                border-bottom: 1px solid #e9ecef;
                color: #2c3e50;
                font-family: Microsoft YaHei;
                font-size: 14px;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #e9ecef;
                color: #2c3e50;
                border-radius: 4px;
            }
        """)

        layout.addWidget(question_label)
        layout.addWidget(self.register_question)

        # 安全问题答案
        answer_label = QtWidgets.QLabel('答案:')
        self.register_answer = QtWidgets.QLineEdit()
        self.register_answer.setPlaceholderText('请输入安全问题答案')

        # 为答案标签添加上边距
        answer_label.setContentsMargins(0, 15, 0, 0)  # 上边距15像素
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

    def closeEvent(self, event):
        """重写关闭事件，确认退出"""
        reply = QtWidgets.QMessageBox.question(
            self,
            '确认退出',
            '您确定要退出系统吗？',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()  # 接受关闭事件
        else:
            event.ignore()  # 忽略关闭事件