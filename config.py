"""Application configuration and NVIDIA NIM model catalog."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


@dataclass(frozen=True)
class ModelConfig:
    """Metadata and supported parameters for an NVIDIA NIM chat endpoint."""

    model_id: str
    display_name: str
    provider: str
    description: str
    supports_temperature: bool = True
    min_temperature: float = 0.0
    max_temperature: float = 1.0
    default_temperature: float = 0.2
    supports_top_p: bool = True
    default_top_p: float = 0.7
    max_tokens_short: int = 512
    max_tokens_detailed: int = 2048


# Curated NVIDIA API Catalog chat models (https://build.nvidia.com/)
CURATED_CHAT_MODELS: tuple[ModelConfig, ...] = (
    ModelConfig(
        model_id="nvidia/nemotron-3.5-bolt-e4b",
        display_name="NVIDIA Nemotron 3.5 Bolt (E4B)",
        provider="NVIDIA",
        description="Ultra-fast, lightweight Nemotron model for responsive everyday chat and RAG.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    ),
    ModelConfig(
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        display_name="NVIDIA Nemotron 3 Ultra (550B)",
        provider="NVIDIA",
        description="Flagship Nemotron reasoning & chat model for complex analysis, RAG, and coding.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=600,
        max_tokens_detailed=2048,
    ),
    ModelConfig(
        model_id="nvidia/nemotron-3.5-lightning-30b-a3b",
        display_name="NVIDIA Nemotron 3.5 Lightning (30B)",
        provider="NVIDIA",
        description="High-speed MoE Nemotron model optimized for low-latency conversation.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    ),
    ModelConfig(
        model_id="nvidia/nemotron-3.5-super-vl-120b-a12b",
        display_name="NVIDIA Nemotron 3.5 Super VL (120B)",
        provider="NVIDIA",
        description="High-capability Nemotron 3.5 Super model for deep analysis and structured synthesis.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    ),
    ModelConfig(
        model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        display_name="NVIDIA Nemotron 3 Nano Reasoning (30B)",
        provider="NVIDIA",
        description="Reasoning-focused Nemotron Nano model.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    ),
    ModelConfig(
        model_id="nvidia/llama-3.1-nemotron-70b-instruct",
        display_name="NVIDIA Llama 3.1 Nemotron 70B Instruct",
        provider="NVIDIA",
        description="General-purpose Nemotron instruct model tuned for helpful, accurate responses.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.3,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    ),
    ModelConfig(
        model_id="nvidia/llama-3.3-nemotron-super-49b-v1",
        display_name="NVIDIA Llama 3.3 Nemotron Super (49B)",
        provider="NVIDIA",
        description="Balanced Nemotron Super model for reasoning, synthesis, and RAG.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    ),
    ModelConfig(
        model_id="meta/llama-3.3-70b-instruct",
        display_name="Meta Llama 3.3 70B Instruct",
        provider="Meta",
        description="Strong multilingual general assistant hosted on NVIDIA NIM.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    ),
    ModelConfig(
        model_id="meta/llama-3.1-8b-instruct",
        display_name="Meta Llama 3.1 8B Instruct",
        provider="Meta",
        description="Lightweight, fast Llama model for everyday Q&A.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=1536,
    ),
    ModelConfig(
        model_id="mistralai/mistral-large-2-instruct",
        display_name="Mistral Large 2 Instruct",
        provider="Mistral AI",
        description="High-capability Mistral endpoint for reasoning and multilingual tasks.",
        supports_temperature=True,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=True,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    ),
)

CURATED_MODEL_MAP: dict[str, ModelConfig] = {m.model_id: m for m in CURATED_CHAT_MODELS}

# Embedding models available on NVIDIA NIM (in order of preference)
CURATED_EMBED_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3.5-embed-1b",
    "nvidia/nemotron-3.5-embed-8b",
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
    "nvidia/embed-qa-4",
    "snowflake/arctic-embed-l",
)

# Substrings indicating non-chat endpoints returned by /v1/models
NON_CHAT_MODEL_KEYWORDS: tuple[str, ...] = (
    "embed",
    "rerank",
    "parse",
    "guard",
    "safety",
    "reward",
    "transcribe",
    "asr",
    "tts",
    "whisper",
    "parakeet",
    "canary",
    "riva",
    "diffusion",
    "flux",
    "sdxl",
    "stable-diffusion",
    "trellis",
    "cosmos",
    "bionemo",
    "alphafold",
    "esmfold",
    "diffdock",
    "molmim",
    "genmol",
    "usd",
    "paddleocr",
    "ocdrnet",
    "deplot",
    "paligemma",
    "grounding-dino",
    "vila",
    "kosmos",
    "streampetr",
    "gliner",
    "pii",
    "detector",
    "calibration",
    "ising",
    "nemoguard",
    "topic-control",
    "nvclip",
)


def _resolve_path(raw_path: str, default_rel: str) -> Path:
    candidate = Path(raw_path or default_rel)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def get_env(name: str, default: str = "") -> str:
    """Read an environment variable trimmed of whitespace."""
    return os.getenv(name, default).strip()


def normalize_api_key(raw_key: str) -> str:
    """Strip accidental 'Bearer ' prefix from API keys."""
    cleaned = (raw_key or "").strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()
    return cleaned


def normalize_base_url(raw_url: str) -> str:
    """Normalize an OpenAI-compatible NVIDIA base URL (strip /chat/completions if pasted)."""
    cleaned = (raw_url or "https://integrate.api.nvidia.com/v1").strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")]
    return cleaned or "https://integrate.api.nvidia.com/v1"


class Settings:
    """Runtime settings loaded from environment variables."""

    NVIDIA_API_KEY: str = normalize_api_key(get_env("NVIDIA_API_KEY"))
    NVIDIA_BASE_URL: str = normalize_base_url(get_env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    NVIDIA_MODEL: str = get_env("NVIDIA_MODEL", CURATED_CHAT_MODELS[0].model_id)
    NVIDIA_EMBED_MODEL: str = get_env("NVIDIA_EMBED_MODEL", CURATED_EMBED_MODELS[0])

    TAVILY_API_KEY: str = normalize_api_key(get_env("TAVILY_API_KEY"))

    MAX_PDF_SIZE_MB: int = int(get_env("MAX_PDF_SIZE_MB", "20") or "20")
    PDF_CHUNK_SIZE: int = 900
    PDF_CHUNK_OVERLAP: int = 180
    PDF_TOP_K: int = 5
    MAX_HISTORY_MESSAGES: int = 16

    SQLITE_DB_PATH: Path = _resolve_path(get_env("SQLITE_DB_PATH", "database/chatbot.db"), "database/chatbot.db")
    UPLOAD_DIR: Path = _resolve_path(get_env("UPLOAD_DIR", "uploaded_files"), "uploaded_files")
    ASSETS_DIR: Path = BASE_DIR / "assets"

    @classmethod
    def reload(cls) -> None:
        """Reload .env values if environment variables changed."""
        load_dotenv(BASE_DIR / ".env", override=True)
        cls.NVIDIA_API_KEY = normalize_api_key(get_env("NVIDIA_API_KEY"))
        cls.NVIDIA_BASE_URL = normalize_base_url(
            get_env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        )
        cls.NVIDIA_MODEL = get_env("NVIDIA_MODEL", CURATED_CHAT_MODELS[0].model_id)
        cls.NVIDIA_EMBED_MODEL = get_env("NVIDIA_EMBED_MODEL", CURATED_EMBED_MODELS[0])
        cls.TAVILY_API_KEY = normalize_api_key(get_env("TAVILY_API_KEY"))
        cls.MAX_PDF_SIZE_MB = int(get_env("MAX_PDF_SIZE_MB", "20") or "20")
        cls.SQLITE_DB_PATH = _resolve_path(
            get_env("SQLITE_DB_PATH", "database/chatbot.db"), "database/chatbot.db"
        )
        cls.UPLOAD_DIR = _resolve_path(get_env("UPLOAD_DIR", "uploaded_files"), "uploaded_files")


def is_chat_model_id(model_id: str) -> bool:
    """Return True if an endpoint model ID appears to be a chat/completion model."""
    lower = model_id.lower()
    return not any(keyword in lower for keyword in NON_CHAT_MODEL_KEYWORDS)


def build_model_config(model_id: str) -> ModelConfig:
    """Return known ModelConfig or infer conservative supported parameters for a model ID."""
    if model_id in CURATED_MODEL_MAP:
        return CURATED_MODEL_MAP[model_id]

    lower = model_id.lower()
    provider = model_id.split("/")[0].upper() if "/" in model_id else "NVIDIA"
    short_name = model_id.split("/")[-1]
    display_name = f"{provider}: {short_name}"

    # Certain reasoning-only endpoints (like OpenAI o-series style or DeepSeek-R1) may restrict temperature
    is_fixed_temp_reasoning = any(tag in lower for tag in ("-r1", "o1-", "o3-"))

    return ModelConfig(
        model_id=model_id,
        display_name=display_name,
        provider=provider,
        description=f"NVIDIA NIM endpoint ({model_id}).",
        supports_temperature=not is_fixed_temp_reasoning,
        min_temperature=0.0,
        max_temperature=1.0,
        default_temperature=0.2,
        supports_top_p=not is_fixed_temp_reasoning,
        default_top_p=0.7,
        max_tokens_short=512,
        max_tokens_detailed=2048,
    )
