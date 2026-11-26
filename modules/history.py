"""Conversation history management service backed by SQLite."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from config import Settings
from database.db import Database
from utils.helpers import generate_chat_title


class ChatHistoryManager:
    """High-level interface for creating, loading, updating, and deleting conversations."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def create_conversation(self, title: str = "New Chat", model: str | None = None) -> dict[str, Any]:
        conv_id = str(uuid.uuid4())
        chosen_model = model or Settings.NVIDIA_MODEL
        return self.db.create_conversation(conv_id=conv_id, title=title, model=chosen_model)

    def list_conversations(self) -> list[dict[str, Any]]:
        return self.db.list_conversations()

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        return self.db.get_conversation(conv_id)

    def get_messages(self, conv_id: str) -> list[dict[str, Any]]:
        return self.db.get_messages(conv_id)

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        conv = self.db.get_conversation(conv_id)
        if conv and role == "user" and conv["title"] == "New Chat":
            self.db.update_conversation_title(conv_id, generate_chat_title(content))
        return self.db.add_message(
            conv_id=conv_id,
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata,
        )

    def clear_conversation(self, conv_id: str) -> None:
        self.db.clear_messages(conv_id)
        self.db.update_conversation_title(conv_id, "New Chat")

    def delete_conversation(self, conv_id: str) -> None:
        file_paths = self.db.delete_conversation(conv_id)
        self._cleanup_unreferenced_files(file_paths)

    def update_model(self, conv_id: str, model: str) -> None:
        self.db.update_conversation_model(conv_id, model)

    def save_document(
        self,
        conv_id: str,
        filename: str,
        file_path: str,
        file_hash: str,
        page_count: int,
        chunk_count: int,
    ) -> dict[str, Any]:
        return self.db.add_document(
            conv_id=conv_id,
            filename=filename,
            file_path=file_path,
            file_hash=file_hash,
            page_count=page_count,
            chunk_count=chunk_count,
        )

    def get_documents(self, conv_id: str) -> list[dict[str, Any]]:
        return self.db.get_documents(conv_id)

    def clear_documents(self, conv_id: str) -> None:
        file_paths = self.db.delete_documents(conv_id)
        self._cleanup_unreferenced_files(file_paths)

    def _cleanup_unreferenced_files(self, file_paths: list[str]) -> None:
        for raw_path in file_paths:
            if not raw_path:
                continue
            if not self.db.is_file_path_referenced(raw_path):
                path = Path(raw_path)
                if path.exists() and path.is_file():
                    try:
                        path.unlink()
                    except OSError:
                        pass
