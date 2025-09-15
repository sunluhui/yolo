import sys
from PyQt5 import QtWidgets
from database import init_database
from ui_login import LoginWindow
from ui_main import MainWindow


def main():
    # 初始化数据库
    init_database()

    # 创建应用
    app = QtWidgets.QApplication(sys.argv)

    # 显示登录窗口
    login_window = LoginWindow()
    login_window.show()

    if app.exec_() == 0 and login_window.current_user:
        # 登录成功，显示主窗口
        user_id, username = login_window.current_user
        main_window = MainWindow(user_id, username)
        main_window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()