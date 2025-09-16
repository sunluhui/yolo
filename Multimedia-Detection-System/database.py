# database.py
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


# 初始化数据库表
create_table()