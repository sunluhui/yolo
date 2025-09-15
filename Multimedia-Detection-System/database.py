import sqlite3
import hashlib
from datetime import datetime


def get_db_connection():
    """创建并返回数据库连接"""
    conn = sqlite3.connect('users.db')
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database():
    """初始化数据库表"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
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
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    conn.commit()
    conn.close()


def hash_password(password):
    """对密码进行SHA-256哈希处理"""
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    """注册新用户"""
    hashed_password = hash_password(password)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    """验证用户登录信息"""
    hashed_password = hash_password(password)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username FROM users WHERE username = ? AND password = ?",
        (username, hashed_password)
    )
    user = cursor.fetchone()
    conn.close()

    return user


def add_detection_record(user_id, detection_type, source_path, result_path):
    """添加检测记录到数据库"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO detection_records 
        (user_id, detection_type, source_path, result_path) 
        VALUES (?, ?, ?, ?)""",
        (user_id, detection_type, source_path, result_path)
    )
    conn.commit()
    conn.close()