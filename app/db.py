import sqlite3
from pathlib import Path
import json
import uuid

DB_PATH = Path(__file__).parent.parent / "chatbot.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() :
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id text PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            created_at text DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            question TEXT,
            sources TEXT,
            answer TEXT,
            top_score REAL,
            unanswerable INTEGER,
            created_at TEXT DEFAULT  (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS unanswered (
            id TEXT PRIMARY KEY,
            history_id TEXT REFERENCES history(id),
            question TEXT,
            top_score REAL,
            reason TEXT, 
            status TEXT DEFAULT ('pending'),
            created_at TEXT DEFAULT  (datetime('now', 'localtime'))
        );
    """)
    conn.commit()
    conn.close()

def save_history(user_id, question, answer, sources, top_score, unanswerable) -> str:
    id = uuid.uuid4().hex
    
    conn = get_conn()

    conn.execute(
        "INSERT INTO history (id, user_id, question, answer, sources, top_score, unanswerable) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (id, user_id, question, answer, json.dumps(sources, ensure_ascii=False), top_score, int(unanswerable))
    )
    conn.commit()
    conn.close()
    return id

def save_unanswered(history_id, question, top_score, reason) -> str:
    id = uuid.uuid4().hex

    conn = get_conn()

    conn.execute(
        "INSERT INTO unanswered (id, history_id, question, top_score, reason) VALUES (?, ?, ?, ?, ?)",
        (id, history_id, question, top_score, reason)
    )
    conn.commit()
    conn.close()
    return id

def get_history():
    conn = get_conn()

    rows = conn.execute(
        "SELECT * FROM history ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_unanswered():
    conn = get_conn()

    rows = conn.execute(
        "SELECT * FROM unanswered"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def create_user(username, password_hash, role) -> str:
    id = uuid.uuid4().hex

    conn = get_conn()
    
    conn.execute(
        "INSERT INTO users (id, username, password_hash, role) VALUES (?,?,?,?)",
        (id, username, password_hash, role)
    )
    conn.commit()
    conn.close()
    return id

def get_user_by_username(username) -> dict | None:
    conn = get_conn()

    row = conn.execute(
        "SELECT * FROM users where username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None