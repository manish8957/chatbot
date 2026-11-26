"""SQLite database initialization and low-level CRUD operations."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import Settings
from utils.helpers import now_iso


class Database:
    """Lightweight SQLite persistence manager for conversations, messages, and PDFs."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else Settings.SQLITE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create required tables and indexes if they do not exist."""
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    uploaded_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv_id
                    ON messages(conversation_id, id);

                CREATE INDEX IF NOT EXISTS idx_documents_conv_id
                    ON documents(conversation_id, id);
                """
            )

    # ---- Conversations ----

    def create_conversation(self, conv_id: str, title: str, model: str) -> dict[str, Any]:
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, title, model, ts, ts),
            )
        return {
            "id": conv_id,
            "title": title,
            "model": model,
            "created_at": ts,
            "updated_at": ts,
        }

    def list_conversations(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.model, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC, c.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, title, model, created_at, updated_at FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_conversation_title(self, conv_id: str, title: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now_iso(), conv_id),
            )

    def update_conversation_model(self, conv_id: str, model: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET model = ?, updated_at = ? WHERE id = ?",
                (model, now_iso(), conv_id),
            )

    def delete_conversation(self, conv_id: str) -> list[str]:
        """Delete a conversation and return associated uploaded file paths for cleanup."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT file_path FROM documents WHERE conversation_id = ?",
                (conv_id,),
            ).fetchall()
            file_paths = [row["file_path"] for row in rows]
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        return file_paths

    # ---- Messages ----

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ts = timestamp or now_iso()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, role, content, ts, metadata_json),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (ts, conv_id),
            )
            msg_id = cursor.lastrowid

        return {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": role,
            "content": content,
            "timestamp": ts,
            "metadata": metadata or {},
        }

    def get_messages(self, conv_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, timestamp, metadata_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conv_id,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            meta: dict[str, Any] = {}
            if row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    meta = {}
            results.append(
                {
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "metadata": meta,
                }
            )
        return results

    def clear_messages(self, conv_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now_iso(), conv_id),
            )

    # ---- Documents ----

    def add_document(
        self,
        conv_id: str,
        filename: str,
        file_path: str,
        file_hash: str,
        page_count: int,
        chunk_count: int,
    ) -> dict[str, Any]:
        ts = now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (
                    conversation_id, filename, file_path, file_hash,
                    page_count, chunk_count, uploaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (conv_id, filename, file_path, file_hash, page_count, chunk_count, ts),
            )
            doc_id = cursor.lastrowid
        return {
            "id": doc_id,
            "conversation_id": conv_id,
            "filename": filename,
            "file_path": file_path,
            "file_hash": file_hash,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "uploaded_at": ts,
        }

    def get_documents(self, conv_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, filename, file_path, file_hash,
                       page_count, chunk_count, uploaded_at
                FROM documents
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conv_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_documents(self, conv_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT file_path FROM documents WHERE conversation_id = ?",
                (conv_id,),
            ).fetchall()
            paths = [row["file_path"] for row in rows]
            conn.execute("DELETE FROM documents WHERE conversation_id = ?", (conv_id,))
        return paths

    def is_file_path_referenced(self, file_path: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE file_path = ? LIMIT 1",
                (file_path,),
            ).fetchone()
        return row is not None
