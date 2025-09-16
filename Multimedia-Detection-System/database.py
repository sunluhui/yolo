import sqlite3
import hashlib
import os
from datetime import datetime
from config import Config


class DatabaseManager:
    def __init__(self, db_path=Config.DB_PATH):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """创建并返回数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_database(self):
        """初始化数据库表"""
        # 创建结果目录
        os.makedirs(Config.IMAGE_RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.VIDEO_RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.CAMERA_RESULTS_DIR, exist_ok=True)

        conn = self.get_connection()
        cursor = conn.cursor()

        # 创建用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # 创建检测记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            detection_type TEXT NOT NULL,
            source_path TEXT,
            result_path TEXT,
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        ''')

        # 创建索引以提高查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_records_user_id ON detection_records(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_records_detection_time ON detection_records(detection_time)')

        # 插入默认管理员用户
        try:
            hashed_password = self.hash_password('admin123')
            hashed_answer = self.hash_password('blue')
            cursor.execute(
                "INSERT INTO users (username, password, security_question, security_answer) VALUES (?, ?, ?, ?)",
                ('admin', hashed_password, '你最喜欢的颜色是什么？', hashed_answer)
            )
        except sqlite3.IntegrityError:
            pass  # 用户已存在

        conn.commit()
        conn.close()

    def hash_password(self, password):
        """对密码进行SHA-256哈希处理"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password, security_question, security_answer):
        """注册新用户"""
        hashed_password = self.hash_password(password)
        hashed_answer = self.hash_password(security_answer)

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, security_question, security_answer) VALUES (?, ?, ?, ?)",
                (username, hashed_password, security_question, hashed_answer)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def verify_user(self, username, password):
        """验证用户登录信息"""
        hashed_password = self.hash_password(password)

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username FROM users WHERE username = ? AND password = ?",
            (username, hashed_password)
        )
        user = cursor.fetchone()
        conn.close()

        return user

    def get_security_question(self, username):
        """获取用户的安全问题"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT security_question FROM users WHERE username = ?",
            (username,)
        )
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def verify_security_answer(self, username, answer):
        """验证安全问题答案"""
        hashed_answer = self.hash_password(answer)

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE username = ? AND security_answer = ?",
            (username, hashed_answer)
        )
        result = cursor.fetchone()
        conn.close()

        return result is not None

    def reset_password(self, username, new_password):
        """重置用户密码"""
        hashed_password = self.hash_password(new_password)

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password = ? WHERE username = ?",
            (hashed_password, username)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success

    def add_detection_record(self, user_id, detection_type, source_path, result_path):
        """添加检测记录到数据库"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO detection_records 
            (user_id, detection_type, source_path, result_path) 
            VALUES (?, ?, ?, ?)""",
            (user_id, detection_type, source_path, result_path)
        )
        conn.commit()
        conn.close()

    def get_detection_records(self, user_id):
        """获取用户的检测记录，并转换为本地时间"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 使用SQLite的datetime函数转换时区
        cursor.execute(
            """SELECT detection_type, source_path, result_path, 
               datetime(detection_time, 'localtime') as local_detection_time
            FROM detection_records WHERE user_id = ? ORDER BY detection_time DESC""",
            (user_id,)
        )

        records = cursor.fetchall()
        conn.close()
        return records