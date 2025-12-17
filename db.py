import sqlite3, pathlib, json
DB_PATH = pathlib.Path(__file__).with_name("portfolio.db")

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                risk_mode  TEXT    NOT NULL,
                weights    TEXT    NOT NULL,
                timeframe  TEXT    NOT NULL,
                preferences TEXT
            );
        """)