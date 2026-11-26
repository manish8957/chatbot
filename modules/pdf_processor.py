"""PDF validation, text extraction via PyPDF, scanned-PDF detection, and chunking."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from config import Settings
from utils.helpers import compute_sha256, safe_filename, validate_pdf_file


class PDFProcessingError(Exception):
    """Raised when a PDF file cannot be read or validated."""


class ScannedPDFError(PDFProcessingError):
    """Raised when a PDF appears to be scanned/image-only without extractable text."""


@dataclass
class PDFChunk:
    """A text chunk extracted from a specific PDF file and page number."""

    text: str
    filename: str
    page_number: int
    chunk_index: int


@dataclass
class ExtractedPDF:
    """Result of extracting and chunking a PDF document."""

    filename: str
    saved_path: str
    file_hash: str
    page_count: int
    chunks: list[PDFChunk]


def split_text_with_overlap(
    text: str,
    chunk_size: int = Settings.PDF_CHUNK_SIZE,
    chunk_overlap: int = Settings.PDF_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks preferring sentence/paragraph boundaries."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    text_len = len(cleaned)
    step_floor = max(100, chunk_size - chunk_overlap)

    while start < text_len:
        end = min(text_len, start + chunk_size)
        if end < text_len:
            # Prefer breaking on sentence boundary or whitespace in the trailing 30%
            window_start = start + int(chunk_size * 0.65)
            split_at = -1
            for sep in (". ", "? ", "! ", "; ", " "):
                pos = cleaned.rfind(sep, window_start, end)
                if pos != -1:
                    split_at = pos + (0 if sep == " " else 1)
                    break
            if split_at > start:
                end = split_at

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break
        next_start = max(start + step_floor, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def process_uploaded_pdf(
    filename: str,
    file_bytes: bytes,
    conversation_id: str,
    upload_dir: Path | None = None,
    max_size_mb: int | None = None,
) -> ExtractedPDF:
    """Validate, persist, extract text page-by-page, detect scanned PDFs, and chunk."""
    resolved_max_mb = max_size_mb or Settings.MAX_PDF_SIZE_MB
    is_valid, reason = validate_pdf_file(filename, file_bytes, max_size_mb=resolved_max_mb)
    if not is_valid:
        raise PDFProcessingError(reason)

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise PDFProcessingError(f"Could not parse PDF '{filename}': {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise PDFProcessingError(
                    f"'{filename}' is password-protected. Please upload an unencrypted PDF."
                )
        except PDFProcessingError:
            raise
        except Exception as exc:
            raise PDFProcessingError(
                f"'{filename}' is encrypted and cannot be read without a password."
            ) from exc

    page_count = len(reader.pages)
    if page_count == 0:
        raise PDFProcessingError(f"'{filename}' contains zero pages.")

    page_texts: list[tuple[int, str]] = []
    total_chars = 0
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        normalized = " ".join(raw_text.split())
        if normalized:
            total_chars += len(normalized)
            page_texts.append((page_idx, raw_text))

    # Scanned / image-only PDF detection
    if total_chars < 25:
        raise ScannedPDFError(
            f"Unsupported document: '{filename}' appears to be a scanned or image-only PDF "
            "(no extractable text found). OCR is not enabled in this app—please upload a "
            "text-based PDF."
        )

    clean_name = safe_filename(filename)
    file_hash = compute_sha256(file_bytes)
    target_dir = Path(upload_dir) if upload_dir else Settings.UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_file = target_dir / f"{conversation_id[:8]}_{file_hash[:8]}_{clean_name}"
    saved_file.write_bytes(file_bytes)

    chunks: list[PDFChunk] = []
    chunk_counter = 0
    for page_num, raw_text in page_texts:
        for piece in split_text_with_overlap(raw_text):
            chunks.append(
                PDFChunk(
                    text=piece,
                    filename=clean_name,
                    page_number=page_num,
                    chunk_index=chunk_counter,
                )
            )
            chunk_counter += 1

    if not chunks:
        raise ScannedPDFError(
            f"Unsupported document: could not extract readable text chunks from '{filename}'."
        )

    return ExtractedPDF(
        filename=clean_name,
        saved_path=str(saved_file),
        file_hash=file_hash,
        page_count=page_count,
        chunks=chunks,
    )
