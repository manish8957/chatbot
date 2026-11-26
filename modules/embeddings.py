"""NVIDIA NIM embeddings and FAISS vector store for PDF RAG retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from config import CURATED_EMBED_MODELS, Settings
from modules.chat import create_nvidia_client
from modules.pdf_processor import PDFChunk
from utils.helpers import sanitize_error_message


class EmbeddingError(Exception):
    """User-safe exception raised when embedding generation or vector search fails."""


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


def _embed_batch(
    client: OpenAI,
    texts: list[str],
    model: str,
    input_type: str,
) -> list[list[float]]:
    """Request embeddings from NVIDIA NIM, handling asymmetric input_type cleanly."""
    try:
        response = client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"},
        )
    except BadRequestError:
        # Fallback if a specific endpoint rejects extra_body parameters
        response = client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
        )
    return [item.embedding for item in response.data]


def generate_embeddings(
    texts: list[str],
    input_type: str = "passage",
    embed_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, str]:
    """Generate L2-normalized embedding matrix for a list of texts using NVIDIA NIM."""
    if not texts:
        raise EmbeddingError("No text provided for embedding generation.")

    try:
        client = create_nvidia_client(api_key=api_key, base_url=base_url, timeout=60.0)
    except Exception as exc:
        raise EmbeddingError(sanitize_error_message(exc)) from exc

    preferred = embed_model or Settings.NVIDIA_EMBED_MODEL
    candidates = [preferred, *[m for m in CURATED_EMBED_MODELS if m != preferred]]

    last_error: Exception | None = None
    for candidate_model in candidates:
        try:
            all_vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                all_vectors.extend(
                    _embed_batch(client, batch, model=candidate_model, input_type=input_type)
                )
            matrix = np.array(all_vectors, dtype="float32")
            return _normalize_vectors(matrix), candidate_model
        except AuthenticationError as exc:
            raise EmbeddingError(
                "Invalid NVIDIA API key while generating embeddings (401)."
            ) from exc
        except RateLimitError as exc:
            raise EmbeddingError(
                "NVIDIA embeddings rate limit reached (429). Please try again shortly."
            ) from exc
        except (APITimeoutError, APIConnectionError) as exc:
            raise EmbeddingError(
                "Network timeout or connection issue while calling NVIDIA embeddings API."
            ) from exc
        except APIStatusError as exc:
            # Try next embedding model candidate if 400/404/410 (unsupported or retired)
            last_error = exc
            if exc.status_code in (400, 404, 410):
                continue
            raise EmbeddingError(
                f"NVIDIA embeddings API error ({exc.status_code}): {sanitize_error_message(exc)}"
            ) from exc
        except Exception as exc:
            last_error = exc
            break

    raise EmbeddingError(
        f"Failed to generate embeddings via NVIDIA NIM: {sanitize_error_message(last_error or 'Unknown error')}"
    )


class FAISSVectorStore:
    """Per-conversation FAISS vector index paired with chunk page metadata."""

    def __init__(self, embed_model: str | None = None) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[PDFChunk] = []
        self.embed_model: str = embed_model or Settings.NVIDIA_EMBED_MODEL

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0 or not self.chunks

    def add_chunks(
        self,
        new_chunks: list[PDFChunk],
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> int:
        """Embed PDF chunks as passages and insert them into the FAISS index."""
        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        vectors, used_model = generate_embeddings(
            texts=texts,
            input_type="passage",
            embed_model=self.embed_model,
            api_key=api_key,
            base_url=base_url,
        )
        self.embed_model = used_model

        dim = vectors.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        elif self.index.d != dim:
            # Reset if embedding dimension changed across models
            self.index = faiss.IndexFlatIP(dim)
            self.chunks = []

        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int = Settings.PDF_TOP_K,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> list[tuple[PDFChunk, float]]:
        """Retrieve top-k relevant PDF chunks for a user question."""
        if self.is_empty or not query.strip():
            return []

        query_vec, _ = generate_embeddings(
            texts=[query.strip()],
            input_type="query",
            embed_model=self.embed_model,
            api_key=api_key,
            base_url=base_url,
        )

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)

        hits: list[tuple[PDFChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                hits.append((self.chunks[idx], float(score)))
        return hits

    @staticmethod
    def format_retrieved_context(hits: list[tuple[PDFChunk, float]]) -> tuple[str, list[dict[str, Any]]]:
        """Format retrieved chunks for the LLM prompt and UI citation pills."""
        if not hits:
            return "", []

        prompt_blocks: list[str] = []
        citations: list[dict[str, Any]] = []
        seen_pages: set[tuple[str, int]] = set()

        for rank, (chunk, score) in enumerate(hits, start=1):
            prompt_blocks.append(
                f"[Excerpt {rank} | File: {chunk.filename} | Page {chunk.page_number} | Relevance: {score:.2f}]\n"
                f"{chunk.text}"
            )
            key = (chunk.filename, chunk.page_number)
            if key not in seen_pages:
                seen_pages.add(key)
                citations.append(
                    {
                        "filename": chunk.filename,
                        "page": chunk.page_number,
                        "snippet": chunk.text[:180] + ("..." if len(chunk.text) > 180 else ""),
                    }
                )

        return "\n\n".join(prompt_blocks), citations

    def save(self, conversation_id: str, base_dir: Path | None = None) -> None:
        """Persist FAISS index and chunk metadata to disk for conversation reload."""
        if self.is_empty:
            return
        target_dir = Path(base_dir) if base_dir else Settings.UPLOAD_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        index_path = target_dir / f"{conversation_id}.faiss"
        meta_path = target_dir / f"{conversation_id}.chunks.json"

        faiss.write_index(self.index, str(index_path))
        payload = {
            "embed_model": self.embed_model,
            "chunks": [asdict(c) for c in self.chunks],
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, conversation_id: str, base_dir: Path | None = None) -> FAISSVectorStore | None:
        """Load a persisted FAISS vector store for a conversation if present."""
        target_dir = Path(base_dir) if base_dir else Settings.UPLOAD_DIR
        index_path = target_dir / f"{conversation_id}.faiss"
        meta_path = target_dir / f"{conversation_id}.chunks.json"
        if not index_path.exists() or not meta_path.exists():
            return None

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            store = cls(embed_model=payload.get("embed_model"))
            store.index = faiss.read_index(str(index_path))
            store.chunks = [PDFChunk(**item) for item in payload.get("chunks", [])]
            return store
        except Exception:
            return None

    @staticmethod
    def delete_persisted(conversation_id: str, base_dir: Path | None = None) -> None:
        """Remove persisted FAISS index and chunk metadata for a conversation."""
        target_dir = Path(base_dir) if base_dir else Settings.UPLOAD_DIR
        for suffix in (".faiss", ".chunks.json"):
            path = target_dir / f"{conversation_id}{suffix}"
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
