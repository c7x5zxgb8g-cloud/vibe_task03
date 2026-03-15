"""SQLite database layer for leads tracking system."""

import os
import sqlite3
import json
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)

_BASE_DIR = os.path.dirname(__file__)
_DB_PATH = os.path.join(_BASE_DIR, "leads.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    wechat_user_id  TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL DEFAULT '',
    group_chat_id   TEXT NOT NULL DEFAULT '',
    group_chat_name TEXT NOT NULL DEFAULT '',
    first_seen_at   TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    note            TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_customers_wechat_user_id ON customers(wechat_user_id);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_id          TEXT UNIQUE,
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    content         TEXT NOT NULL,
    msg_type        TEXT NOT NULL DEFAULT 'text',
    group_chat_id   TEXT NOT NULL DEFAULT '',
    received_at     TEXT NOT NULL,
    processed       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_customer_id ON messages(customer_id);
CREATE INDEX IF NOT EXISTS idx_messages_processed ON messages(processed);
CREATE INDEX IF NOT EXISTS idx_messages_received_at ON messages(received_at);

CREATE TABLE IF NOT EXISTS intent_analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id      INTEGER NOT NULL REFERENCES messages(id),
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    intent_level    TEXT NOT NULL,
    intent_score    REAL NOT NULL DEFAULT 0,
    intent_summary  TEXT NOT NULL DEFAULT '',
    keywords        TEXT NOT NULL DEFAULT '[]',
    raw_response    TEXT NOT NULL DEFAULT '',
    analyzed_at     TEXT NOT NULL,
    notified        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_intent_customer ON intent_analyses(customer_id);
CREATE INDEX IF NOT EXISTS idx_intent_level ON intent_analyses(intent_level);

CREATE TABLE IF NOT EXISTS follow_ups (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         INTEGER NOT NULL REFERENCES customers(id),
    intent_analysis_id  INTEGER REFERENCES intent_analyses(id),
    reply_type          TEXT NOT NULL DEFAULT 'auto',
    status              TEXT NOT NULL DEFAULT 'pending',
    generated_message   TEXT NOT NULL DEFAULT '',
    attachment_files    TEXT NOT NULL DEFAULT '[]',
    admin_note          TEXT NOT NULL DEFAULT '',
    target_user_id      TEXT NOT NULL DEFAULT '',
    target_group_id     TEXT NOT NULL DEFAULT '',
    confirmed_at        TEXT,
    sent_at             TEXT,
    error_message       TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_followup_customer ON follow_ups(customer_id);
CREATE INDEX IF NOT EXISTS idx_followup_status ON follow_ups(status);

CREATE TABLE IF NOT EXISTS team_members (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    wechat_id       TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name       TEXT NOT NULL,
    file_type       TEXT NOT NULL DEFAULT '',
    file_size       INTEGER NOT NULL DEFAULT 0,
    storage_path    TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'uploaded',
    total_pages     INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT NOT NULL DEFAULT '',
    uploaded_at     TEXT NOT NULL,
    processed_at    TEXT,
    published_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_kb_doc_status ON kb_documents(status);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id      INTEGER NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL DEFAULT 0,
    text             TEXT NOT NULL DEFAULT '',
    page_number      INTEGER NOT NULL DEFAULT 0,
    category         TEXT NOT NULL DEFAULT '',
    keywords         TEXT NOT NULL DEFAULT '',
    summary          TEXT NOT NULL DEFAULT '',
    importance       TEXT NOT NULL DEFAULT '中',
    related_products TEXT NOT NULL DEFAULT '',
    manually_edited  INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kb_chunk_doc ON kb_chunks(document_id);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(_SCHEMA_SQL)
    # Migrations for follow_ups table
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(follow_ups)").fetchall()]
        if "attachment_files" not in cols:
            conn.execute("ALTER TABLE follow_ups ADD COLUMN attachment_files TEXT NOT NULL DEFAULT '[]'")
            conn.commit()
            logger.info("Migrated follow_ups: added attachment_files column")
        if "reply_type" not in cols:
            conn.execute("ALTER TABLE follow_ups ADD COLUMN reply_type TEXT NOT NULL DEFAULT 'auto'")
            conn.commit()
            logger.info("Migrated follow_ups: added reply_type column")
    except Exception as e:
        logger.warning(f"Migration check failed: {e}")

    # Seed team members from config
    from config import Config
    if Config.WECHAT_TEAM_IDS:
        for wid in Config.WECHAT_TEAM_IDS:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO team_members (wechat_id, name, created_at) VALUES (?, ?, ?)",
                    (wid, wid, datetime.now().isoformat())
                )
            except Exception:
                pass
        conn.commit()
    conn.close()
    logger.info(f"Database initialized at {_DB_PATH}")


# ── Customer CRUD ──────────────────────────────────────────────


def upsert_customer(wechat_user_id: str, name: str, group_chat_id: str = "",
                    group_chat_name: str = "") -> int:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM customers WHERE wechat_user_id = ?",
            (wechat_user_id,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE customers SET name=?, group_chat_id=?, group_chat_name=?, last_seen_at=? WHERE id=?",
                (name, group_chat_id, group_chat_name, now, row["id"])
            )
            conn.commit()
            return row["id"]
        else:
            cur = conn.execute(
                "INSERT INTO customers (wechat_user_id, name, group_chat_id, group_chat_name, first_seen_at, last_seen_at) VALUES (?,?,?,?,?,?)",
                (wechat_user_id, name, group_chat_id, group_chat_name, now, now)
            )
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()


def get_customer(customer_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM customers WHERE id=?", (customer_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_customers(limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM customers ORDER BY last_seen_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Message CRUD ───────────────────────────────────────────────


def insert_message(customer_id: int, content: str, msg_type: str = "text",
                   group_chat_id: str = "", msg_id: str | None = None,
                   received_at: str | None = None) -> int | None:
    now = received_at or datetime.now().isoformat()
    conn = get_connection()
    try:
        if msg_id:
            existing = conn.execute("SELECT id FROM messages WHERE msg_id=?", (msg_id,)).fetchone()
            if existing:
                return None  # duplicate
        cur = conn.execute(
            "INSERT INTO messages (msg_id, customer_id, content, msg_type, group_chat_id, received_at) VALUES (?,?,?,?,?,?)",
            (msg_id, customer_id, content, msg_type, group_chat_id, now)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_unprocessed_messages(limit: int = 20) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM messages WHERE processed=0 AND msg_type='text' ORDER BY received_at ASC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_message_processed(message_id: int):
    conn = get_connection()
    try:
        conn.execute("UPDATE messages SET processed=1 WHERE id=?", (message_id,))
        conn.commit()
    finally:
        conn.close()


def get_messages_by_customer(customer_id: int, limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM messages WHERE customer_id=? ORDER BY received_at DESC LIMIT ?",
            (customer_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_messages(customer_id: int | None = None, group_chat_id: str | None = None,
                  limit: int = 100, offset: int = 0) -> list[dict]:
    conn = get_connection()
    try:
        sql = """
            SELECT m.*, c.name as customer_name, c.wechat_user_id
            FROM messages m JOIN customers c ON m.customer_id = c.id
            WHERE 1=1
        """
        params: list = []
        if customer_id is not None:
            sql += " AND m.customer_id=?"
            params.append(customer_id)
        if group_chat_id:
            sql += " AND m.group_chat_id=?"
            params.append(group_chat_id)
        sql += " ORDER BY m.received_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Intent Analysis CRUD ───────────────────────────────────────


def insert_intent_analysis(message_id: int, customer_id: int, intent_level: str,
                           intent_score: float, intent_summary: str,
                           keywords: str = "[]", raw_response: str = "") -> int:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO intent_analyses (message_id, customer_id, intent_level, intent_score, intent_summary, keywords, raw_response, analyzed_at) VALUES (?,?,?,?,?,?,?,?)",
            (message_id, customer_id, intent_level, intent_score, intent_summary, keywords, raw_response, now)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def mark_intent_notified(analysis_id: int):
    conn = get_connection()
    try:
        conn.execute("UPDATE intent_analyses SET notified=1 WHERE id=?", (analysis_id,))
        conn.commit()
    finally:
        conn.close()


def get_high_intent_leads(intent_level: str | None = None,
                          limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_connection()
    try:
        sql = """
            SELECT ia.*, c.name as customer_name, c.wechat_user_id, c.group_chat_name,
                   m.content as message_content, m.received_at as message_time
            FROM intent_analyses ia
            JOIN customers c ON ia.customer_id = c.id
            JOIN messages m ON ia.message_id = m.id
            WHERE 1=1
        """
        params: list = []
        if intent_level:
            sql += " AND ia.intent_level=?"
            params.append(intent_level)
        else:
            sql += " AND ia.intent_level IN ('high','medium')"
        sql += " ORDER BY ia.intent_score DESC, ia.analyzed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Follow-up CRUD ─────────────────────────────────────────────


def create_follow_up(customer_id: int, intent_analysis_id: int | None = None,
                     target_user_id: str = "", target_group_id: str = "",
                     reply_type: str = "auto") -> int:
    """Create a follow-up record.

    reply_type: 'auto' for normal KB auto-replies, 'follow_up' for sales follow-up (needs admin approval).
    """
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO follow_ups (customer_id, intent_analysis_id, reply_type, target_user_id, target_group_id, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (customer_id, intent_analysis_id, reply_type, target_user_id, target_group_id, "pending", now, now)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_follow_up(follow_up_id: int, **kwargs):
    if not kwargs:
        return
    kwargs["updated_at"] = datetime.now().isoformat()
    conn = get_connection()
    try:
        set_parts = [f"{k}=?" for k in kwargs]
        values = list(kwargs.values()) + [follow_up_id]
        conn.execute(
            f"UPDATE follow_ups SET {', '.join(set_parts)} WHERE id=?",
            values
        )
        conn.commit()
    finally:
        conn.close()


def get_follow_up(follow_up_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM follow_ups WHERE id=?", (follow_up_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_follow_ups(status: str | None = None, reply_type: str | None = None,
                   limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_connection()
    try:
        sql = """
            SELECT f.*, c.name as customer_name, c.wechat_user_id, c.group_chat_name
            FROM follow_ups f JOIN customers c ON f.customer_id = c.id
            WHERE 1=1
        """
        params: list = []
        if status:
            sql += " AND f.status=?"
            params.append(status)
        if reply_type:
            sql += " AND f.reply_type=?"
            params.append(reply_type)
        sql += " ORDER BY f.updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def has_recent_wechat_card(customer_id: int | None = None,
                          group_chat_id: str | None = None,
                          hours: int = 12) -> bool:
    """Check if a wechat card was already sent recently.

    Dedup dimensions:
      - If group_chat_id is provided: check if ANY reply in that group within the
        last `hours` already contains a wechat card (group-level dedup).
      - Otherwise fall back to customer-level dedup.
    """
    from datetime import timedelta
    conn = get_connection()
    try:
        cutoff_str = (datetime.now() - timedelta(hours=hours)).isoformat()

        if group_chat_id:
            # Group-level: join with customers to match group_chat_id
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM follow_ups f
                   JOIN customers c ON f.customer_id = c.id
                   WHERE c.group_chat_id = ?
                     AND f.created_at >= ?
                     AND f.attachment_files LIKE '%wechat_card%'""",
                (group_chat_id, cutoff_str)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM follow_ups WHERE customer_id=? AND created_at>=? AND attachment_files LIKE '%wechat_card%'",
                (customer_id, cutoff_str)
            ).fetchone()
        return row["cnt"] > 0
    finally:
        conn.close()


def get_pending_follow_ups() -> list[dict]:
    """Get follow-ups that are ready to be sent (status='message_generated')."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT f.*, c.name as customer_name, c.wechat_user_id,
                   c.group_chat_id, c.group_chat_name
            FROM follow_ups f JOIN customers c ON f.customer_id = c.id
            WHERE f.status = 'message_generated'
            ORDER BY f.updated_at ASC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Dashboard Stats ────────────────────────────────────────────


def get_dashboard_stats() -> dict:
    conn = get_connection()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        today_msgs = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE received_at >= ?", (today,)
        ).fetchone()["cnt"]
        high_leads = conn.execute(
            "SELECT COUNT(*) as cnt FROM intent_analyses WHERE intent_level IN ('high','medium')"
        ).fetchone()["cnt"]
        pending_followups = conn.execute(
            "SELECT COUNT(*) as cnt FROM follow_ups WHERE status IN ('pending','confirmed','message_generated')"
        ).fetchone()["cnt"]
        sent_followups = conn.execute(
            "SELECT COUNT(*) as cnt FROM follow_ups WHERE status='sent'"
        ).fetchone()["cnt"]
        total_customers = conn.execute(
            "SELECT COUNT(*) as cnt FROM customers"
        ).fetchone()["cnt"]
        total_messages = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages"
        ).fetchone()["cnt"]
        return {
            "today_messages": today_msgs,
            "high_intent_leads": high_leads,
            "pending_follow_ups": pending_followups,
            "sent_follow_ups": sent_followups,
            "total_customers": total_customers,
            "total_messages": total_messages,
        }
    finally:
        conn.close()


# ── Knowledge Base Document CRUD ──────────────────────────────


def insert_kb_document(file_name: str, file_type: str, file_size: int,
                       storage_path: str) -> int:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO kb_documents (file_name, file_type, file_size, storage_path, status, uploaded_at) VALUES (?,?,?,?,?,?)",
            (file_name, file_type, file_size, storage_path, "uploaded", now)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_kb_document(doc_id: int, **kwargs):
    if not kwargs:
        return
    conn = get_connection()
    try:
        set_parts = [f"{k}=?" for k in kwargs]
        values = list(kwargs.values()) + [doc_id]
        conn.execute(
            f"UPDATE kb_documents SET {', '.join(set_parts)} WHERE id=?",
            values
        )
        conn.commit()
    finally:
        conn.close()


def get_kb_document(doc_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_kb_documents(status: str | None = None, limit: int = 50,
                      offset: int = 0) -> list[dict]:
    conn = get_connection()
    try:
        sql = "SELECT * FROM kb_documents WHERE 1=1"
        params: list = []
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY uploaded_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_kb_document(doc_id: int):
    """Delete a document and all its chunks (CASCADE)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM kb_documents WHERE id=?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


# ── Knowledge Base Chunk CRUD ─────────────────────────────────


def insert_kb_chunks(document_id: int, chunks_data: list[dict]):
    """Bulk insert chunks for a document."""
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        for c in chunks_data:
            conn.execute(
                "INSERT INTO kb_chunks (document_id, chunk_index, text, page_number, category, keywords, summary, importance, related_products, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    c.get("chunk_index", 0),
                    c.get("text", ""),
                    c.get("page_number", 0),
                    c.get("category", ""),
                    c.get("keywords", ""),
                    c.get("summary", ""),
                    c.get("importance", "中"),
                    c.get("related_products", ""),
                    now,
                )
            )
        conn.commit()
    finally:
        conn.close()


def list_kb_chunks(document_id: int) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM kb_chunks WHERE document_id=? ORDER BY chunk_index ASC",
            (document_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_kb_chunk(chunk_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM kb_chunks WHERE id=?", (chunk_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_kb_chunk(chunk_id: int, **kwargs):
    if not kwargs:
        return
    conn = get_connection()
    try:
        set_parts = [f"{k}=?" for k in kwargs]
        values = list(kwargs.values()) + [chunk_id]
        conn.execute(
            f"UPDATE kb_chunks SET {', '.join(set_parts)} WHERE id=?",
            values
        )
        conn.commit()
    finally:
        conn.close()


# ── Team Members CRUD ─────────────────────────────────────────


def get_team_member_ids() -> set[str]:
    """Return set of all team member wechat IDs."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT wechat_id FROM team_members").fetchall()
        return {r["wechat_id"] for r in rows}
    finally:
        conn.close()


def list_team_members() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM team_members ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def add_team_member(wechat_id: str, name: str = "") -> int | None:
    """Add a team member. Returns id or None if duplicate."""
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM team_members WHERE wechat_id=?", (wechat_id,)).fetchone()
        if existing:
            return None
        cur = conn.execute(
            "INSERT INTO team_members (wechat_id, name, created_at) VALUES (?,?,?)",
            (wechat_id, name or wechat_id, datetime.now().isoformat())
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def remove_team_member(member_id: int):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM team_members WHERE id=?", (member_id,))
        conn.commit()
    finally:
        conn.close()


# ── Message Management ────────────────────────────────────────


def delete_message(message_id: int) -> bool:
    """Delete a single message by ID. Also removes related intent_analyses and follow_ups.

    Returns True if a message was deleted, False if not found.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM messages WHERE id=?", (message_id,)).fetchone()
        if not row:
            return False
        # Delete related intent analyses and their follow-ups
        analysis_ids = [
            r["id"] for r in conn.execute(
                "SELECT id FROM intent_analyses WHERE message_id=?", (message_id,)
            ).fetchall()
        ]
        if analysis_ids:
            placeholders = ",".join("?" * len(analysis_ids))
            conn.execute(
                f"DELETE FROM follow_ups WHERE intent_analysis_id IN ({placeholders})",
                analysis_ids,
            )
            conn.execute(
                f"DELETE FROM intent_analyses WHERE id IN ({placeholders})",
                analysis_ids,
            )
        conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def delete_all_messages() -> int:
    """Delete all messages and related intent_analyses and follow_ups.

    Returns the number of messages deleted.
    """
    conn = get_connection()
    try:
        # Delete follow_ups linked to intent_analyses
        conn.execute(
            "DELETE FROM follow_ups WHERE intent_analysis_id IN (SELECT id FROM intent_analyses)"
        )
        # Delete all intent analyses
        conn.execute("DELETE FROM intent_analyses")
        # Count and delete messages
        count = conn.execute("SELECT COUNT(*) as cnt FROM messages").fetchone()["cnt"]
        conn.execute("DELETE FROM messages")
        conn.commit()
        return count
    finally:
        conn.close()


def get_kb_document_stats() -> dict:
    conn = get_connection()
    try:
        total_docs = conn.execute("SELECT COUNT(*) as cnt FROM kb_documents").fetchone()["cnt"]
        published_docs = conn.execute(
            "SELECT COUNT(*) as cnt FROM kb_documents WHERE status='published'"
        ).fetchone()["cnt"]
        processing_docs = conn.execute(
            "SELECT COUNT(*) as cnt FROM kb_documents WHERE status='processing'"
        ).fetchone()["cnt"]
        ready_docs = conn.execute(
            "SELECT COUNT(*) as cnt FROM kb_documents WHERE status='ready'"
        ).fetchone()["cnt"]
        total_chunks = conn.execute("SELECT COUNT(*) as cnt FROM kb_chunks").fetchone()["cnt"]
        return {
            "total_documents": total_docs,
            "published_documents": published_docs,
            "processing_documents": processing_docs,
            "ready_documents": ready_docs,
            "total_chunks": total_chunks,
        }
    finally:
        conn.close()


# ── System Settings CRUD ─────────────────────────────────────


def get_setting(key: str, default: str = "") -> str:
    """Get a single setting value by key."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM system_settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


def set_setting(key: str, value: str):
    """Upsert a single setting."""
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO system_settings (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, now)
        )
        conn.commit()
    finally:
        conn.close()


def get_all_settings() -> dict:
    """Get all settings as a dict."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, value FROM system_settings").fetchall()
        return {r["key"]: r["value"] for r in rows}
    finally:
        conn.close()
