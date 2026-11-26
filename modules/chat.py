"""NVIDIA NIM API client, model discovery, prompt assembly, and streaming chat."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from config import (
    CURATED_CHAT_MODELS,
    CURATED_MODEL_MAP,
    ModelConfig,
    Settings,
    build_model_config,
    is_chat_model_id,
    normalize_api_key,
    normalize_base_url,
)
from utils.helpers import is_valid_api_key, sanitize_error_message


class NvidiaAPIError(Exception):
    """User-safe error raised when an NVIDIA NIM API request fails."""


def create_nvidia_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
) -> OpenAI:
    """Create an OpenAI-compatible client pointed at the NVIDIA NIM endpoint."""
    resolved_key = normalize_api_key(api_key if api_key is not None else Settings.NVIDIA_API_KEY)
    resolved_url = normalize_base_url(base_url if base_url is not None else Settings.NVIDIA_BASE_URL)

    if not is_valid_api_key(resolved_key, expected_prefix="nvapi-"):
        raise NvidiaAPIError(
            "NVIDIA API key is missing or still set to a placeholder. "
            "Add a valid NVIDIA_API_KEY to your .env file."
        )

    return OpenAI(
        base_url=resolved_url,
        api_key=resolved_key,
        timeout=timeout,
        max_retries=1,
    )


def check_nvidia_connection(
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[bool, str, list[str]]:
    """Verify NVIDIA NIM connectivity and return (ok, status_message, available_model_ids)."""
    resolved_key = (api_key if api_key is not None else Settings.NVIDIA_API_KEY).strip()
    if not is_valid_api_key(resolved_key, expected_prefix="nvapi-"):
        return False, "Missing NVIDIA_API_KEY in .env", []

    try:
        client = create_nvidia_client(api_key=resolved_key, base_url=base_url, timeout=12.0)
        models_page = client.models.list()
        model_ids = sorted({m.id for m in models_page.data if getattr(m, "id", None)})
        return True, f"Connected ({len(model_ids)} catalog models)", model_ids
    except AuthenticationError:
        return False, "Invalid NVIDIA API key (401 Unauthorized)", []
    except RateLimitError:
        return False, "NVIDIA API rate limit reached (429)", []
    except APITimeoutError:
        return False, "NVIDIA API connection timed out", []
    except APIConnectionError:
        return False, "Unable to reach NVIDIA NIM endpoint", []
    except APIStatusError as exc:
        return False, f"NVIDIA API error ({exc.status_code})", []
    except Exception as exc:
        return False, f"Connection check failed: {sanitize_error_message(exc)}", []


def get_available_chat_models(discovered_ids: list[str] | None = None) -> list[ModelConfig]:
    """Return ordered list of supported NVIDIA NIM chat models."""
    models: dict[str, ModelConfig] = {}

    # Always prioritize configured default model if valid
    configured_default = Settings.NVIDIA_MODEL
    if configured_default:
        models[configured_default] = build_model_config(configured_default)

    if discovered_ids:
        # Include curated models that exist in live catalog first, then remaining live chat models
        discovered_set = set(discovered_ids)
        for curated in CURATED_CHAT_MODELS:
            if curated.model_id in discovered_set:
                models[curated.model_id] = curated
        for model_id in discovered_ids:
            if model_id not in models and is_chat_model_id(model_id):
                models[model_id] = build_model_config(model_id)
    else:
        for curated in CURATED_CHAT_MODELS:
            models.setdefault(curated.model_id, curated)

    return list(models.values())


def build_system_prompt(
    response_style: str = "short",
    web_context: str | None = None,
    pdf_context: str | None = None,
) -> str:
    """Construct the system prompt enforcing accuracy, conciseness, and grounded citations."""
    style_instruction = (
        "Provide short, accurate, and easy-to-understand answers by default. "
        "Keep responses concise and directly to the point unless the user explicitly asks for more detail."
        if response_style == "short"
        else "Provide thorough, well-structured, detailed explanations with clear steps, reasoning, and examples where helpful."
    )

    sections = [
        "You are a helpful, accurate, and intelligent AI assistant powered by NVIDIA NIM.",
        style_instruction,
        (
            "Never fabricate facts, URLs, search results, or document citations. "
            "If you are uncertain or lack sufficient information, say so clearly."
        ),
    ]

    if web_context:
        sections.append(
            "WEB SEARCH CONTEXT (retrieved live via Tavily):\n"
            f"{web_context}\n\n"
            "Instructions for Web Search:\n"
            "- Base time-sensitive or factual claims on the web results above.\n"
            "- Include inline markdown source links [Source Title](url) referencing ONLY URLs present in the search results.\n"
            "- Do not invent any sources or URLs."
        )

    if pdf_context:
        sections.append(
            "UPLOADED PDF DOCUMENT CONTEXT:\n"
            f"{pdf_context}\n\n"
            "Instructions for PDF Q&A:\n"
            "- Answer the user's question using the extracted PDF passages above.\n"
            "- Cite exact page numbers (e.g., `(Page 3)`) and filenames wherever possible.\n"
            "- If the uploaded PDF does not contain the answer, state that clearly rather than guessing."
        )

    return "\n\n".join(sections)


def _format_api_exception(exc: Exception) -> NvidiaAPIError:
    """Translate OpenAI/NVIDIA SDK exceptions into clean user-facing messages."""
    if isinstance(exc, AuthenticationError):
        return NvidiaAPIError(
            "Authentication failed with NVIDIA NIM (401). Please verify your NVIDIA_API_KEY in `.env`."
        )
    if isinstance(exc, RateLimitError):
        return NvidiaAPIError(
            "NVIDIA NIM rate limit reached (429). Please wait a moment and try again."
        )
    if isinstance(exc, APITimeoutError):
        return NvidiaAPIError(
            "The NVIDIA NIM request timed out. Try a shorter prompt or another model."
        )
    if isinstance(exc, APIConnectionError):
        return NvidiaAPIError(
            "Could not connect to the NVIDIA NIM API. Check your internet connection and NVIDIA_BASE_URL."
        )
    if isinstance(exc, BadRequestError):
        detail = sanitize_error_message(exc)
        return NvidiaAPIError(
            f"The selected NVIDIA endpoint rejected the request parameters: {detail}"
        )
    if isinstance(exc, APIStatusError):
        detail = sanitize_error_message(exc)
        return NvidiaAPIError(f"NVIDIA NIM returned status {exc.status_code}: {detail}")
    return NvidiaAPIError(f"Unexpected NVIDIA API error: {sanitize_error_message(exc)}")


def stream_chat_completion(
    messages: list[dict[str, str]],
    model_id: str,
    temperature: float | None = 0.2,
    response_style: str = "short",
    web_context: str | None = None,
    pdf_context: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Generator[str, None, None]:
    """Stream text chunks from an NVIDIA NIM chat completion endpoint."""
    client = create_nvidia_client(api_key=api_key, base_url=base_url)
    model_cfg = CURATED_MODEL_MAP.get(model_id) or build_model_config(model_id)

    system_prompt = build_system_prompt(
        response_style=response_style,
        web_context=web_context,
        pdf_context=pdf_context,
    )

    # Keep recent conversation turns for context without exceeding token budgets
    trimmed_history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages[-Settings.MAX_HISTORY_MESSAGES :]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    payload_messages = [{"role": "system", "content": system_prompt}, *trimmed_history]

    max_tokens = (
        model_cfg.max_tokens_short
        if response_style == "short"
        else model_cfg.max_tokens_detailed
    )

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": payload_messages,
        "max_tokens": max_tokens,
        "stream": True,
    }

    if model_cfg.supports_temperature and temperature is not None:
        clamped_temp = max(
            model_cfg.min_temperature,
            min(model_cfg.max_temperature, float(temperature)),
        )
        kwargs["temperature"] = clamped_temp

    if model_cfg.supports_top_p:
        kwargs["top_p"] = model_cfg.default_top_p

    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
    except BadRequestError as exc:
        # Some catalog endpoints disallow optional sampling parameters like top_p; retry cleanly without them.
        if "top_p" in kwargs or "temperature" in kwargs:
            retry_kwargs = {
                "model": model_id,
                "messages": payload_messages,
                "max_tokens": max_tokens,
                "stream": True,
            }
            try:
                stream = client.chat.completions.create(**retry_kwargs)
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
                return
            except Exception as retry_exc:
                raise _format_api_exception(retry_exc) from retry_exc
        raise _format_api_exception(exc) from exc
    except Exception as exc:
        raise _format_api_exception(exc) from exc
