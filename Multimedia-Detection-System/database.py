import sqlite3
import hashlib
from datetime import datetime

import sqlite3
import bcrypt
from datetime import datetime

DATABASE = 'users.db'


def create_table():
    """创建用户表，包含安全问题字段"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            security_question1 TEXT NOT NULL DEFAULT '你最喜欢的颜色是什么？',
            security_answer1 TEXT NOT NULL,
            security_question2 TEXT NOT NULL DEFAULT '你的爱好是什么？',
            security_answer2 TEXT NOT NULL,
            security_question3 TEXT NOT NULL DEFAULT '你的生日是哪天？(格式:YYYY-MM-DD)',
            security_answer3 TEXT NOT NULL,
            reset_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def register_user(username, password, security_answers):
    """注册新用户"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # 检查用户名是否已存在
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        conn.close()
        return False

    # 加密密码
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # 插入新用户
    cursor.execute(
        "INSERT INTO users (username, password, security_answer1, security_answer2, security_answer3) VALUES (?, ?, ?, ?, ?)",
        (username, hashed_password, security_answers[0], security_answers[1], security_answers[2])
    )
    conn.commit()
    conn.close()
    return True


def verify_user(username, password):
    """验证用户登录"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username=?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
        return True
    return False


def get_user_by_username(username):
    """根据用户名获取用户信息"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def update_password(username, new_password):
    """更新用户密码"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute("UPDATE users SET password=? WHERE username=?", (hashed_password, username))
    conn.commit()
    conn.close()
    return True


def get_security_answers(username):
    """获取用户的安全问题答案"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT security_answer1, security_answer2, security_answer3 FROM users WHERE username=?",
        (username,)
    )
    answers = cursor.fetchone()
    conn.close()
    return answers


def set_reset_token(username, token):
    """设置密码重置令牌"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET reset_token=? WHERE username=?", (token, username))
    conn.commit()
    conn.close()


def verify_reset_token(username, token):
    """验证密码重置令牌"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT reset_token FROM users WHERE username=?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result and result[0] == token


def clear_reset_token(username):
    """清除密码重置令牌"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET reset_token=NULL WHERE username=?", (username,))
    conn.commit()
    conn.close()


# 初始化数据库表
create_table()
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