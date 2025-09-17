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
    # 界面配置
    WINDOW_TITLE = "小目标检测系统"
    WINDOW_SIZE = (1200, 800)
    THEME_STYLE = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        /* 添加标题样式 */
        QLabel#mainTitle {
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            padding: 20px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2c3e50);
            border-radius: 10px;
            margin: 10px;
            border: 2px solid #2980b9;
            text-align: center;
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
            font-size: 14px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            border-bottom: 2px solid #007acc;
            font-weight: bold;
        }
        QPushButton {
            background-color: #007acc;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
        QLabel {
            color: #333333;
            font-size: 14px;
        }
        QLineEdit {
            padding: 8px;
            border: 1px solid #cccccc;
            border-radius: 4px;
            font-size: 14px;
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
            padding: 8px;
            border: 1px solid #cccccc;
            font-weight: bold;
        }
    """