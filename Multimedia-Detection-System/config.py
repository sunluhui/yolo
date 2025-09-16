# 系统配置
import os


class Config:
    # 数据库配置
    DB_PATH = 'users.db'

    # 结果保存路径
    RESULTS_DIR = 'results'
    IMAGE_RESULTS_DIR = os.path.join(RESULTS_DIR, 'images')
    VIDEO_RESULTS_DIR = os.path.join(RESULTS_DIR, 'videos')
    CAMERA_RESULTS_DIR = os.path.join(RESULTS_DIR, 'camera')

    # 模型配置
    MODEL_PATH = 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = 0.5

    # 界面配置
    WINDOW_TITLE = "小目标检测系统"
    WINDOW_SIZE = (1200, 800)
    THEME_STYLE = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: #ffffff;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            padding: 8px 16px;
            border: 1px solid #cccccc;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            border-bottom: 2px solid #007acc;
        }
        QPushButton {
            background-color: #007acc;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
        QLabel {
            color: #333333;
        }
        QLineEdit {
            padding: 6px;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        QTableWidget {
            gridline-color: #cccccc;
            background-color: #ffffff;
            alternate-background-color: #f5f5f5;
        }
        QTableWidget::item:selected {
            background-color: #007acc;
            color: white;
        }
        QHeaderView::section {
            background-color: #e0e0e0;
            padding: 6px;
            border: 1px solid #cccccc;
        }
    """