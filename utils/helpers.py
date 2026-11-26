"""Shared utility helpers for validation, formatting, and secret redaction."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

PLACEHOLDER_FRAGMENTS = (
    "your-nvidia-api-key",
    "your-tavily-api-key",
    "your_api_key",
    "changeme",
    "replace-me",
    "example",
)

SECRET_PATTERNS = (
    re.compile(r"nvapi-[A-Za-z0-9_\-]+"),
    re.compile(r"tvly-[A-Za-z0-9_\-]+"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", re.IGNORECASE),
)


def is_valid_api_key(api_key: str | None, expected_prefix: str = "") -> bool:
    """Check whether an API key is present and not an obvious placeholder."""
    if not api_key:
        return False
    cleaned = api_key.strip()
    if len(cleaned) < 10:
        return False
    lower = cleaned.lower()
    if any(fragment in lower for fragment in PLACEHOLDER_FRAGMENTS):
        return False
    if expected_prefix and not cleaned.startswith(expected_prefix):
        # Still allow non-standard keys if user uses a custom gateway, unless placeholder
        return True
    return True


def sanitize_error_message(error: Exception | str) -> str:
    """Redact potential API keys or Bearer tokens from error text before showing to users."""
    text = str(error)
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(r"\1[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text.strip()


def now_iso() -> str:
    """Return current local timestamp as ISO-8601 string (seconds precision)."""
    return datetime.now().isoformat(timespec="seconds")


def format_timestamp(iso_str: str | None) -> str:
    """Format an ISO timestamp for compact display in chat bubbles and sidebar."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        today = datetime.now().date()
        if dt.date() == today:
            return dt.strftime("%I:%M %p").lstrip("0")
        return dt.strftime("%b %d, %I:%M %p").replace(" 0", " ")
    except ValueError:
        return iso_str


def truncate_text(text: str, max_chars: int = 44) -> str:
    """Collapse whitespace and truncate text cleanly."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def generate_chat_title(first_prompt: str) -> str:
    """Generate a concise conversation title from the first user prompt."""
    title = truncate_text(first_prompt, max_chars=42)
    return title or "New Chat"


def validate_pdf_file(filename: str, file_bytes: bytes, max_size_mb: int = 20) -> tuple[bool, str]:
    """Validate uploaded PDF extension, non-emptiness, size limit, and PDF magic header."""
    if not filename or not filename.lower().endswith(".pdf"):
        return False, "Only `.pdf` files are supported."

    if not file_bytes:
        return False, "The uploaded PDF file is empty."

    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        return (
            False,
            f"PDF size ({size_mb:.1f} MB) exceeds the {max_size_mb} MB limit.",
        )

    if not file_bytes.lstrip()[:8].startswith(b"%PDF-"):
        return False, "Invalid PDF file: missing standard `%PDF-` header."

    return True, "Valid PDF"


def compute_sha256(data: bytes) -> str:
    """Return SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def safe_filename(filename: str) -> str:
    """Sanitize an uploaded filename to prevent path traversal."""
    base = Path(filename).name
    sanitized = re.sub(r"[^A-Za-z0-9._\- ]+", "_", base).strip()
    return sanitized or "document.pdf"
