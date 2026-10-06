import os
import json
import uuid
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    return conn

def init_db() :
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id text PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            created_at TIMESTAMP(0) DEFAULT (now() AT TIME ZONE 'Asia/Seoul')
        );
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            question TEXT,
            sources TEXT,
            answer TEXT,
            top_score DOUBLE PRECISION,
            unanswerable INTEGER,
            created_at TIMESTAMP(0) DEFAULT (now() AT TIME ZONE 'Asia/Seoul')
        );
        CREATE TABLE IF NOT EXISTS unanswered (
            id TEXT PRIMARY KEY,
            history_id TEXT REFERENCES history(id),
            question TEXT,
            top_score DOUBLE PRECISION,
            reason TEXT, 
            status TEXT DEFAULT ('pending'),
            created_at TIMESTAMP(0) DEFAULT (now() AT TIME ZONE 'Asia/Seoul')
        );
    """)
    conn.commit()
    conn.close()

def save_history(user_id, question, answer, sources, top_score, unanswerable) -> str:
    id = uuid.uuid4().hex
    
    conn = get_conn()

    conn.execute(
        "INSERT INTO history (id, user_id, question, answer, sources, top_score, unanswerable) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (id, user_id, question, answer, json.dumps(sources, ensure_ascii=False), top_score, int(unanswerable))
    )
    conn.commit()
    conn.close()
    return id

def save_unanswered(history_id, question, top_score, reason) -> str:
    id = uuid.uuid4().hex

    conn = get_conn()

    conn.execute(
        "INSERT INTO unanswered (id, history_id, question, top_score, reason) VALUES (%s, %s, %s, %s, %s)",
        (id, history_id, question, top_score, reason)
    )
    conn.commit()
    conn.close()
    return id

def get_history():
    conn = get_conn()

    # users와 JOIN해서 질문한 사람의 아이디(username)도 함께 가져온다
    # LEFT JOIN: 로그인 기능 전에 쌓인 이력(user_id 없음)도 빠지지 않게
    rows = conn.execute(
        """SELECT h.*, u.username
           FROM history h
           LEFT JOIN users u ON u.id = h.user_id
           ORDER BY h.created_at DESC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_unanswered():
    conn = get_conn()

    rows = conn.execute(
        "SELECT * FROM unanswered ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_unanswered_status(unanswered_id: str, status: str) -> bool:
    """미답변 처리 상태 변경 (pending ↔ resolved). 없는 id면 False"""
    conn = get_conn()
    cur = conn.execute(
        "UPDATE unanswered SET status = %s WHERE id = %s",
        (status, unanswered_id)
    )
    updated = cur.rowcount   # 실제로 바뀐 줄 수
    conn.commit()
    conn.close()
    return updated > 0

def create_user(username, password_hash, role) -> str:
    id = uuid.uuid4().hex

    conn = get_conn()
    
    conn.execute(
        "INSERT INTO users (id, username, password_hash, role) VALUES (%s,%s,%s,%s)",
        (id, username, password_hash, role)
    )
    conn.commit()
    conn.close()
    return id

def get_user_by_username(username) -> dict | None:
    conn = get_conn()

    row = conn.execute(
        "SELECT * FROM users where username = %s", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_list() -> dict | None:
    conn = get_conn()

    rows = conn.execute(
        "SELECT username, role, created_at FROM users ORDER BY created_at"
    ).fetchall()
    conn.close()
    return rows