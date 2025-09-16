import sys
import os
from PyQt5 import QtWidgets
from database import DatabaseManager
from yolo_detector import YOLODetector
from ui_login import LoginWindow
from ui_main import MainWindow
from config import Config


def main():
    # 初始化数据库
    db_manager = DatabaseManager()

    # 初始化YOLO检测器
    try:
        detector = YOLODetector()
        print("YOLO检测器初始化成功")
    except Exception as e:
        print(f"YOLO检测器初始化失败: {e}")
        QtWidgets.QMessageBox.critical(None, '错误', f'YOLO检测器初始化失败: {e}')
        return

    # 创建应用
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格

    # 显示登录窗口
    login_window = LoginWindow(db_manager)

    if login_window.exec_() == QtWidgets.QDialog.Accepted and login_window.current_user:
        # 登录成功，显示主窗口
        user_id, username = login_window.current_user
        main_window = MainWindow(user_id, username, db_manager, detector)
        main_window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()