"""Tavily web search integration for current-information retrieval and source citation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tavily import TavilyClient

from config import Settings
from utils.helpers import is_valid_api_key, sanitize_error_message

AUTO_SEARCH_PATTERN = re.compile(
    r"\b("
    r"latest|recent|recently|today|tonight|yesterday|this week|this month|this year|"
    r"current|currently|now|news|update|updates|breaking|live|price|stock|weather|"
    r"forecast|score|standings|winner|won|election|release date|announced|"
    r"2024|2025|2026|2027|who is the current|what happened"
    r")\b",
    re.IGNORECASE,
)


class WebSearchError(Exception):
    """User-safe exception raised when Tavily web search fails."""


@dataclass
class SearchSource:
    """Single verified search result returned by Tavily."""

    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class WebSearchResult:
    """Structured Tavily search output."""

    query: str
    sources: list[SearchSource] = field(default_factory=list)
    answer: str | None = None

    def format_for_prompt(self) -> str:
        """Render search results into a grounded prompt context block."""
        if not self.sources:
            return "No live web search results were returned."

        lines: list[str] = []
        if self.answer:
            lines.append(f"Tavily Summary: {self.answer}\n")

        for idx, src in enumerate(self.sources, start=1):
            snippet = " ".join(src.content.split())
            if len(snippet) > 600:
                snippet = snippet[:597] + "..."
            lines.append(
                f"[{idx}] Title: {src.title}\n"
                f"URL: {src.url}\n"
                f"Excerpt: {snippet}"
            )
        return "\n\n".join(lines)

    def to_metadata_list(self) -> list[dict[str, str]]:
        """Serialize sources for SQLite storage and UI display."""
        return [
            {
                "title": src.title,
                "url": src.url,
                "snippet": " ".join(src.content.split())[:240],
            }
            for src in self.sources
        ]


def check_tavily_status(api_key: str | None = None) -> tuple[bool, str]:
    """Check whether Tavily API key is configured."""
    key = (api_key if api_key is not None else Settings.TAVILY_API_KEY).strip()
    if not is_valid_api_key(key, expected_prefix="tvly-"):
        return False, "Missing TAVILY_API_KEY in .env"
    return True, "Tavily API key configured"


def should_use_web_search(query: str, mode: str) -> bool:
    """Decide whether to invoke Tavily web search based on user mode and query intent."""
    normalized = (mode or "Off").strip().lower()
    if normalized == "on":
        return True
    if normalized == "auto":
        return bool(AUTO_SEARCH_PATTERN.search(query))
    return False


def search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    api_key: str | None = None,
) -> WebSearchResult:
    """Execute a live web search via Tavily and return verified results with URLs."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise WebSearchError("Cannot perform web search with an empty query.")

    resolved_key = (api_key if api_key is not None else Settings.TAVILY_API_KEY).strip()
    if not is_valid_api_key(resolved_key, expected_prefix="tvly-"):
        raise WebSearchError(
            "Tavily API key is missing or still set to a placeholder. "
            "Add TAVILY_API_KEY to your `.env` file to use web search."
        )

    try:
        client = TavilyClient(api_key=resolved_key)
        response: dict[str, Any] = client.search(
            query=cleaned_query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
        )
    except Exception as exc:
        msg = sanitize_error_message(exc)
        lower = msg.lower()
        if "unauthorized" in lower or "invalid api key" in lower or "401" in lower:
            raise WebSearchError(
                "Invalid Tavily API key. Please check TAVILY_API_KEY in `.env`."
            ) from exc
        if "rate limit" in lower or "429" in lower:
            raise WebSearchError(
                "Tavily API rate limit reached. Please wait a moment and try again."
            ) from exc
        raise WebSearchError(f"Web search failed: {msg}") from exc

    sources: list[SearchSource] = []
    for item in response.get("results", []):
        url = (item.get("url") or "").strip()
        title = (item.get("title") or url or "Web Source").strip()
        content = (item.get("content") or "").strip()
        score = float(item.get("score") or 0.0)
        if url:
            sources.append(SearchSource(title=title, url=url, content=content, score=score))

    return WebSearchResult(
        query=cleaned_query,
        sources=sources,
        answer=response.get("answer"),
    )
