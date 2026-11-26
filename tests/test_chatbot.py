"""Automated tests for core chatbot modules and Streamlit UI startup."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest

from config import build_model_config, is_chat_model_id
from database.db import Database
from modules.chat import NvidiaAPIError, build_system_prompt, get_available_chat_models, stream_chat_completion
from modules.embeddings import FAISSVectorStore
from modules.history import ChatHistoryManager
from modules.pdf_processor import PDFChunk, PDFProcessingError, ScannedPDFError, process_uploaded_pdf
from modules.web_search import SearchSource, WebSearchError, WebSearchResult, search_web, should_use_web_search
from utils.helpers import is_valid_api_key, sanitize_error_message, validate_pdf_file


def _build_minimal_text_pdf(pages: list[str]) -> bytes:
    """Create a valid PDF in memory containing selectable text for each page."""
    # Construct minimal raw PDF with Helvetica font and page content streams
    objects: list[bytes] = []

    # 1: Catalog, 2: Pages, 3: Font
    font_obj_num = 3
    page_obj_nums: list[int] = []
    content_obj_nums: list[int] = []

    next_num = 4
    for _ in pages:
        page_obj_nums.append(next_num)
        content_obj_nums.append(next_num + 1)
        next_num += 2

    kids_str = " ".join(f"{num} 0 R" for num in page_obj_nums)
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(pages)} >>\nendobj\n".encode())
    objects.append(
        f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode()
    )

    for text, p_num, c_num in zip(pages, page_obj_nums, content_obj_nums):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_data = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        objects.append(
            f"{p_num} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> /Contents {c_num} 0 R >>\nendobj\n".encode()
        )
        objects.append(
            f"{c_num} 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode()
            + stream_data
            + b"\nendstream\nendobj\n"
        )

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(buf.tell())
        buf.write(obj)

    xref_pos = buf.tell()
    buf.write(f"xref\n0 {len(objects) + 1}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode())

    buf.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return buf.getvalue()


def _build_blank_pdf_bytes() -> bytes:
    """Create a valid PDF with a blank page (simulates scanned/image-only PDF without text)."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_helpers_validation_and_redaction() -> None:
    assert not is_valid_api_key("")
    assert not is_valid_api_key("nvapi-your-nvidia-api-key-here")
    assert is_valid_api_key("nvapi-abc1234567890xyz", expected_prefix="nvapi-")

    raw_err = "Request failed with key nvapi-Secret12345 and Bearer tok_abc999 and tvly-MyKey123"
    cleaned = sanitize_error_message(raw_err)
    assert "Secret12345" not in cleaned
    assert "tok_abc999" not in cleaned
    assert "MyKey123" not in cleaned

    ok, _ = validate_pdf_file("notes.txt", b"%PDF-1.4", max_size_mb=5)
    assert not ok
    ok, _ = validate_pdf_file("bad.pdf", b"not a pdf", max_size_mb=5)
    assert not ok
    ok, _ = validate_pdf_file("big.pdf", b"%PDF-" + b"x" * (2 * 1024 * 1024), max_size_mb=1)
    assert not ok
    ok, _ = validate_pdf_file("good.pdf", b"%PDF-1.4 sample", max_size_mb=5)
    assert ok


def test_sqlite_history_lifecycle(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    history = ChatHistoryManager(db=db)

    conv = history.create_conversation(model="nvidia/nemotron-3-ultra-550b-a55b")
    assert conv["title"] == "New Chat"

    history.add_message(conv["id"], "user", "Explain FAISS vector similarity search")
    history.add_message(
        conv["id"],
        "assistant",
        "FAISS indexes dense vectors for fast nearest-neighbor search.",
        metadata={"used_web_search": False, "used_pdf": True},
    )

    loaded_conv = history.get_conversation(conv["id"])
    assert loaded_conv is not None
    assert loaded_conv["title"].startswith("Explain FAISS")

    messages = history.get_messages(conv["id"])
    assert len(messages) == 2
    assert messages[1]["metadata"]["used_pdf"] is True

    history.clear_conversation(conv["id"])
    assert history.get_messages(conv["id"]) == []
    assert history.get_conversation(conv["id"])["title"] == "New Chat"

    history.delete_conversation(conv["id"])
    assert history.get_conversation(conv["id"]) is None


def test_pdf_processor_text_and_scanned_detection(tmp_path: Path) -> None:
    pdf_bytes = _build_minimal_text_pdf(
        [
            "Page one discusses NVIDIA NIM microservices and TensorRT-LLM.",
            "Page two covers FAISS vector indexing and retrieval augmented generation.",
        ]
    )
    extracted = process_uploaded_pdf(
        filename="architecture.pdf",
        file_bytes=pdf_bytes,
        conversation_id="conv-12345678",
        upload_dir=tmp_path,
    )
    assert extracted.page_count == 2
    assert len(extracted.chunks) >= 2
    assert extracted.chunks[0].page_number == 1
    assert extracted.chunks[1].page_number == 2
    assert Path(extracted.saved_path).exists()

    scanned_bytes = _build_blank_pdf_bytes()
    with pytest.raises(ScannedPDFError):
        process_uploaded_pdf(
            filename="scanned.pdf",
            file_bytes=scanned_bytes,
            conversation_id="conv-12345678",
            upload_dir=tmp_path,
        )

    with pytest.raises(PDFProcessingError):
        process_uploaded_pdf(
            filename="invalid.pdf",
            file_bytes=b"plain text",
            conversation_id="conv-12345678",
            upload_dir=tmp_path,
        )


def test_faiss_vector_store_search_and_persistence(tmp_path: Path) -> None:
    chunks = [
        PDFChunk(text="NVIDIA NIM runs optimized inference containers.", filename="guide.pdf", page_number=1, chunk_index=0),
        PDFChunk(text="Tavily provides real-time web search results.", filename="guide.pdf", page_number=2, chunk_index=1),
    ]

    def fake_embeddings(texts: list[str], input_type: str = "passage", **kwargs):
        vecs = []
        for t in texts:
            if "NVIDIA" in t or "inference" in t:
                vecs.append([1.0, 0.0, 0.0, 0.0])
            else:
                vecs.append([0.0, 1.0, 0.0, 0.0])
        return np.array(vecs, dtype="float32"), "nvidia/llama-3.2-nv-embedqa-1b-v2"

    with patch("modules.embeddings.generate_embeddings", side_effect=fake_embeddings):
        store = FAISSVectorStore()
        added = store.add_chunks(chunks)
        assert added == 2
        store.save("conv-faiss", base_dir=tmp_path)

        reloaded = FAISSVectorStore.load("conv-faiss", base_dir=tmp_path)
        assert reloaded is not None
        hits = reloaded.search("How does NVIDIA inference work?", top_k=1)
        assert len(hits) == 1
        assert hits[0][0].page_number == 1

        ctx, citations = reloaded.format_retrieved_context(hits)
        assert "Page 1" in ctx
        assert citations[0]["filename"] == "guide.pdf"
        assert citations[0]["page"] == 1


def test_web_search_routing_and_error_handling() -> None:
    assert should_use_web_search("What is the latest news on AI today?", "Auto")
    assert not should_use_web_search("Explain binary search in Python", "Auto")
    assert should_use_web_search("Explain binary search", "On")
    assert not should_use_web_search("Latest news today", "Off")

    res = WebSearchResult(
        query="NVIDIA NIM",
        sources=[SearchSource(title="NVIDIA Build", url="https://build.nvidia.com", content="NIM catalog")],
    )
    assert "https://build.nvidia.com" in res.format_for_prompt()

    with pytest.raises(WebSearchError):
        search_web("latest news", api_key="tvly-your-tavily-api-key-here")


def test_chat_model_filtering_and_error_handling() -> None:
    assert is_chat_model_id("nvidia/nemotron-3-ultra-550b-a55b")
    assert not is_chat_model_id("nvidia/llama-3.2-nv-embedqa-1b-v2")
    assert not is_chat_model_id("nvidia/nemotron-parse")

    models = get_available_chat_models(
        ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nv-embedqa-e5-v5", "meta/llama-3.3-70b-instruct"]
    )
    ids = [m.model_id for m in models]
    assert "nvidia/nemotron-3-ultra-550b-a55b" in ids
    assert "meta/llama-3.3-70b-instruct" in ids
    assert "nvidia/nv-embedqa-e5-v5" not in ids

    reasoning_cfg = build_model_config("deepseek-ai/deepseek-r1")
    assert reasoning_cfg.supports_temperature is False

    prompt = build_system_prompt(response_style="short", web_context="[1] Source", pdf_context="Page 1 text")
    assert "short, accurate" in prompt
    assert "WEB SEARCH CONTEXT" in prompt
    assert "UPLOADED PDF DOCUMENT CONTEXT" in prompt

    with pytest.raises(NvidiaAPIError):
        list(
            stream_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model_id="nvidia/nemotron-3-ultra-550b-a55b",
                api_key="nvapi-your-nvidia-api-key-here",
            )
        )


def test_streamlit_app_boots_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-test the full Streamlit app using AppTest without needing external API keys."""
    import streamlit as st
    from config import Settings

    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setattr(Settings, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(Settings, "TAVILY_API_KEY", "")
    monkeypatch.setattr(Settings, "SQLITE_DB_PATH", tmp_path / "app_test.db")
    monkeypatch.setattr(Settings, "UPLOAD_DIR", tmp_path / "uploads")
    st.cache_resource.clear()

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    assert len(at.sidebar) > 0

    # Test theme toggle button interaction
    theme_btn = [b for b in at.sidebar.button if b.label in ("☀️", "🌙")]
    assert len(theme_btn) == 1
    theme_btn[0].click().run()
    assert not at.exception
    st.cache_resource.clear()
