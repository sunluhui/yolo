import sys
import os
from PyQt5 import QtWidgets
from database import DatabaseManager
from yolo_detector import YOLODetector
from ui_login import LoginWindow
from ui_main import MainWindow


def main():
    # 初始化数据库
    db_manager = DatabaseManager()

    # 初始化YOLO检测器
    detector = YOLODetector()

    # 创建应用
    app = QtWidgets.QApplication(sys.argv)

    # 显示登录窗口
    login_window = LoginWindow(db_manager)
    login_window.show()

    if app.exec_() == 0 and login_window.current_user:
        # 登录成功，显示主窗口
        user_id, username = login_window.current_user
        main_window = MainWindow(user_id, username, db_manager, detector)
        main_window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)


if __name__ == '__main__':
    # 创建结果目录
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('results/videos', exist_ok=True)
    os.makedirs('results/camera', exist_ok=True)

    main()