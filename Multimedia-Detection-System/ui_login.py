from PyQt5 import QtWidgets, QtCore, QtGui


class LoginWindow(QtWidgets.QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.current_user = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('小目标检测系统 - 登录')
        self.setGeometry(400, 300, 400, 300)

        # 创建选项卡
        self.tabs = QtWidgets.QTabWidget()

        # 登录标签页
        self.login_tab = QtWidgets.QWidget()
        self.setup_login_tab()
        self.tabs.addTab(self.login_tab, "登录")

        # 注册标签页
        self.register_tab = QtWidgets.QWidget()
        self.setup_register_tab()
        self.tabs.addTab(self.register_tab, "注册")

        # 找回密码标签页
        self.forgot_tab = QtWidgets.QWidget()
        self.setup_forgot_tab()
        self.tabs.addTab(self.forgot_tab, "找回密码")

        # 主布局
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def setup_login_tab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)

        # 标题
        title_label = QtWidgets.QLabel('用户登录')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # 用户名输入
        username_layout = QtWidgets.QHBoxLayout()
        username_label = QtWidgets.QLabel('用户名:')
        username_label.setFixedWidth(80)
        self.login_username = QtWidgets.QLineEdit()
        self.login_username.setPlaceholderText('请输入用户名')
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.login_username)
        layout.addLayout(username_layout)

        # 密码输入
        password_layout = QtWidgets.QHBoxLayout()
        password_label = QtWidgets.QLabel('密码:')
        password_label.setFixedWidth(80)
        self.login_password = QtWidgets.QLineEdit()
        self.login_password.setPlaceholderText('请输入密码')
        self.login_password.setEchoMode(QtWidgets.QLineEdit.Password)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.login_password)
        layout.addLayout(password_layout)

        # 登录按钮
        self.login_btn = QtWidgets.QPushButton('登录')
        self.login_btn.clicked.connect(self.login)
        layout.addWidget(self.login_btn)

        # 状态标签
        self.login_status = QtWidgets.QLabel('')
        self.login_status.setAlignment(QtCore.Qt.AlignCenter)
        self.login_status.setStyleSheet('color: red;')
        layout.addWidget(self.login_status)

        self.login_tab.setLayout(layout)

    def setup_register_tab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)

        # 标题
        title_label = QtWidgets.QLabel('用户注册')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # 用户名输入
        username_layout = QtWidgets.QHBoxLayout()
        username_label = QtWidgets.QLabel('用户名:')
        username_label.setFixedWidth(80)
        self.register_username = QtWidgets.QLineEdit()
        self.register_username.setPlaceholderText('请输入用户名')
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.register_username)
        layout.addLayout(username_layout)

        # 密码输入
        password_layout = QtWidgets.QHBoxLayout()
        password_label = QtWidgets.QLabel('密码:')
        password_label.setFixedWidth(80)
        self.register_password = QtWidgets.QLineEdit()
        self.register_password.setPlaceholderText('请输入密码')
        self.register_password.setEchoMode(QtWidgets.QLineEdit.Password)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.register_password)
        layout.addLayout(password_layout)

        # 确认密码
        confirm_layout = QtWidgets.QHBoxLayout()
        confirm_label = QtWidgets.QLabel('确认密码:')
        confirm_label.setFixedWidth(80)
        self.register_confirm = QtWidgets.QLineEdit()
        self.register_confirm.setPlaceholderText('请再次输入密码')
        self.register_confirm.setEchoMode(QtWidgets.QLineEdit.Password)
        confirm_layout.addWidget(confirm_label)
        confirm_layout.addWidget(self.register_confirm)
        layout.addLayout(confirm_layout)

        # 安全问题
        question_layout = QtWidgets.QHBoxLayout()
        question_label = QtWidgets.QLabel('安全问题:')
        question_label.setFixedWidth(80)
        self.register_question = QtWidgets.QComboBox()
        self.register_question.addItems([
            '你最喜欢的同性或者异性是谁？',
            '你的出生城市是哪里？',
            '你的第一所学校的名字是什么？',
            '你的宠物的名字是什么？',
            '你母亲的名字是什么？'
        ])
        question_layout.addWidget(question_label)
        question_layout.addWidget(self.register_question)
        layout.addLayout(question_layout)

        # 安全问题答案
        answer_layout = QtWidgets.QHBoxLayout()
        answer_label = QtWidgets.QLabel('答案:')
        answer_label.setFixedWidth(80)
        self.register_answer = QtWidgets.QLineEdit()
        self.register_answer.setPlaceholderText('请输入安全问题答案')
        answer_layout.addWidget(answer_label)
        answer_layout.addWidget(self.register_answer)
        layout.addLayout(answer_layout)

        # 注册按钮
        self.register_btn = QtWidgets.QPushButton('注册')
        self.register_btn.clicked.connect(self.register)
        layout.addWidget(self.register_btn)

        # 状态标签
        self.register_status = QtWidgets.QLabel('')
        self.register_status.setAlignment(QtCore.Qt.AlignCenter)
        self.register_status.setStyleSheet('color: red;')
        layout.addWidget(self.register_status)

        self.register_tab.setLayout(layout)

    def setup_forgot_tab(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)

        # 标题
        title_label = QtWidgets.QLabel('找回密码')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # 用户名输入
        username_layout = QtWidgets.QHBoxLayout()
        username_label = QtWidgets.QLabel('用户名:')
        username_label.setFixedWidth(80)
        self.forgot_username = QtWidgets.QLineEdit()
        self.forgot_username.setPlaceholderText('请输入用户名')
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.forgot_username)
        layout.addLayout(username_layout)

        # 获取安全问题按钮
        self.get_question_btn = QtWidgets.QPushButton('获取安全问题')
        self.get_question_btn.clicked.connect(self.get_security_question)
        layout.addWidget(self.get_question_btn)

        # 安全问题显示
        self.forgot_question = QtWidgets.QLabel('')
        self.forgot_question.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.forgot_question)

        # 安全问题答案
        answer_layout = QtWidgets.QHBoxLayout()
        answer_label = QtWidgets.QLabel('答案:')
        answer_label.setFixedWidth(80)
        self.forgot_answer = QtWidgets.QLineEdit()
        self.forgot_answer.setPlaceholderText('请输入安全问题答案')
        answer_layout.addWidget(answer_label)
        answer_layout.addWidget(self.forgot_answer)
        layout.addLayout(answer_layout)

        # 新密码
        new_password_layout = QtWidgets.QHBoxLayout()
        new_password_label = QtWidgets.QLabel('新密码:')
        new_password_label.setFixedWidth(80)
        self.forgot_new_password = QtWidgets.QLineEdit()
        self.forgot_new_password.setPlaceholderText('请输入新密码')
        self.forgot_new_password.setEchoMode(QtWidgets.QLineEdit.Password)
        new_password_layout.addWidget(new_password_label)
        new_password_layout.addWidget(self.forgot_new_password)
        layout.addLayout(new_password_layout)

        # 确认新密码
        confirm_layout = QtWidgets.QHBoxLayout()
        confirm_label = QtWidgets.QLabel('确认密码:')
        confirm_label.setFixedWidth(80)
        self.forgot_confirm = QtWidgets.QLineEdit()
        self.forgot_confirm.setPlaceholderText('请再次输入新密码')
        self.forgot_confirm.setEchoMode(QtWidgets.QLineEdit.Password)
        confirm_layout.addWidget(confirm_label)
        confirm_layout.addWidget(self.forgot_confirm)
        layout.addLayout(confirm_layout)

        # 重置密码按钮
        self.reset_btn = QtWidgets.QPushButton('重置密码')
        self.reset_btn.clicked.connect(self.reset_password)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn)

        # 状态标签
        self.forgot_status = QtWidgets.QLabel('')
        self.forgot_status.setAlignment(QtCore.Qt.AlignCenter)
        self.forgot_status.setStyleSheet('color: red;')
        layout.addWidget(self.forgot_status)

        self.forgot_tab.setLayout(layout)

    def login(self):
        username = self.login_username.text().strip()
        password = self.login_password.text().strip()

        if not username or not password:
            self.login_status.setText('用户名和密码不能为空')
            return

        user = self.db_manager.verify_user(username, password)
        if user:
            self.current_user = user
            self.login_status.setText('登录成功!')
            QtCore.QTimer.singleShot(500, self.accept_login)
        else:
            self.login_status.setText('用户名或密码错误')

    def register(self):
        username = self.register_username.text().strip()
        password = self.register_password.text().strip()
        confirm = self.register_confirm.text().strip()
        question = self.register_question.currentText()
        answer = self.register_answer.text().strip()

        if not username or not password:
            self.register_status.setText('用户名和密码不能为空')
            return

        if len(username) < 3:
            self.register_status.setText('用户名至少3个字符')
            return

        if len(password) < 6:
            self.register_status.setText('密码至少6个字符')
            return

        if password != confirm:
            self.register_status.setText('两次输入的密码不一致')
            return

        if not answer:
            self.register_status.setText('安全问题答案不能为空')
            return

        if self.db_manager.register_user(username, password, question, answer):
            self.register_status.setText('注册成功，请登录')
            # 清空表单
            self.register_username.clear()
            self.register_password.clear()
            self.register_confirm.clear()
            self.register_answer.clear()
        else:
            self.register_status.setText('用户名已存在')

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
            self.forgot_status.setText('用户名不存在')

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
            self.forgot_status.setText('密码重置成功，请登录')
            # 清空表单
            self.forgot_username.clear()
            self.forgot_answer.clear()
            self.forgot_new_password.clear()
            self.forgot_confirm.clear()
            self.forgot_question.setText('')
            self.reset_btn.setEnabled(False)
        else:
            self.forgot_status.setText('密码重置失败')

    def accept_login(self):
        self.close()