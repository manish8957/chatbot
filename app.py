"""NVIDIA NIM Intelligent AI Chatbot — Streamlit Application."""

from __future__ import annotations

import html
import itertools
import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from config import CURATED_MODEL_MAP, ModelConfig, Settings, build_model_config
from modules.chat import (
    NvidiaAPIError,
    check_nvidia_connection,
    get_available_chat_models,
    stream_chat_completion,
)
from modules.embeddings import EmbeddingError, FAISSVectorStore
from modules.history import ChatHistoryManager
from modules.pdf_processor import PDFProcessingError, ScannedPDFError, process_uploaded_pdf
from modules.web_search import (
    WebSearchError,
    check_tavily_status,
    search_web,
    should_use_web_search,
)
from utils.helpers import format_timestamp, now_iso

st.set_page_config(
    page_title="NVIDIA NIM AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_PROMPTS: tuple[dict[str, str], ...] = (
    {
        "title": "Explain a concept clearly",
        "prompt": "Explain how Retrieval-Augmented Generation (RAG) works in 4 concise bullet points.",
        "web_mode": "Off",
    },
    {
        "title": "Search live information",
        "prompt": "What are the latest developments in NVIDIA Nemotron models?",
        "web_mode": "On",
    },
    {
        "title": "Compare engineering tradeoffs",
        "prompt": "Compare FAISS flat index vs IVF index for small-to-medium PDF collections.",
        "web_mode": "Off",
    },
    {
        "title": "Deep technical breakdown",
        "prompt": "Give a detailed explanation of attention mechanisms in transformer models with an example.",
        "web_mode": "Off",
    },
)


# ---------------------------------------------------------------------------
# Cached resources & API probes
# ---------------------------------------------------------------------------


@st.cache_resource
def get_history_manager() -> ChatHistoryManager:
    """Create a singleton SQLite history manager."""
    return ChatHistoryManager()


@st.cache_data(ttl=300, show_spinner=False)
def cached_nvidia_probe(api_key: str, base_url: str) -> tuple[bool, str, list[str]]:
    """Cache NVIDIA NIM connection status and discovered model IDs for 5 minutes."""
    return check_nvidia_connection(api_key=api_key, base_url=base_url)


@st.cache_data(ttl=300, show_spinner=False)
def cached_tavily_probe(api_key: str) -> tuple[bool, str]:
    """Cache Tavily configuration check."""
    return check_tavily_status(api_key=api_key)


# ---------------------------------------------------------------------------
# Theme & Styling
# ---------------------------------------------------------------------------


def inject_theme_css(theme: str) -> None:
    """Apply Dark or Light theme variables + custom stylesheet."""
    if theme == "Light":
        theme_vars = """
        :root {
            --bg-main: #ffffff;
            --bg-sidebar: #f4f6f8;
            --surface-elevated: #ffffff;
            --surface-subtle: #f1f5f9;
            --user-bubble: #eef6e6;
            --assistant-bubble: #f8fafc;
            --text-primary: #0f172a;
            --text-secondary: #334155;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }
        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-main);
            color: var(--text-primary);
        }
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
        }
        [data-testid="stHeader"] {
            background-color: rgba(255, 255, 255, 0.85);
        }
        [data-testid="stBottom"] > div {
            background-color: var(--bg-main);
        }
        """
    else:
        theme_vars = """
        :root {
            --bg-main: #0d1117;
            --bg-sidebar: #161b22;
            --surface-elevated: #161b22;
            --surface-subtle: #1f2630;
            --user-bubble: #18261a;
            --assistant-bubble: #161b22;
            --text-primary: #f0f6fc;
            --text-secondary: #c9d1d9;
            --text-muted: #8b949e;
            --border-color: #30363d;
        }
        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-main);
            color: var(--text-primary);
        }
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
        }
        [data-testid="stHeader"] {
            background-color: rgba(13, 17, 23, 0.85);
        }
        [data-testid="stBottom"] > div {
            background-color: var(--bg-main);
        }
        """

    css_path = Settings.ASSETS_DIR / "style.css"
    custom_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    st.markdown(f"<style>{theme_vars}\n{custom_css}</style>", unsafe_allow_html=True)


def render_copy_button(text: str, key_id: int | str, theme: str) -> None:
    """Render a lightweight client-side clipboard copy button for a chat message."""
    encoded = json.dumps(text)
    fg = "#334155" if theme == "Light" else "#8b949e"
    hover_fg = "#0f172a" if theme == "Light" else "#f0f6fc"
    border = "#cbd5e1" if theme == "Light" else "#30363d"
    bg = "#f8fafc" if theme == "Light" else "#1f2630"

    html = f"""
    <div style="display:flex;justify-content:flex-end;margin:0;padding:0;">
      <button id="copy-btn-{key_id}" onclick="copyMsg_{key_id}()"
        style="cursor:pointer;font-family:Inter,sans-serif;font-size:12px;font-weight:500;
               padding:3px 10px;border-radius:7px;border:1px solid {border};
               background:{bg};color:{fg};transition:all 0.15s ease;"
        onmouseover="this.style.color='{hover_fg}';this.style.borderColor='#76b900';"
        onmouseout="this.style.color='{fg}';this.style.borderColor='{border}';">
        Copy response
      </button>
    </div>
    <script>
      function copyMsg_{key_id}() {{
        const text = {encoded};
        navigator.clipboard.writeText(text).then(() => {{
          const btn = document.getElementById("copy-btn-{key_id}");
          btn.innerText = "Copied ✓";
          btn.style.color = "#76b900";
          setTimeout(() => {{
            btn.innerText = "Copy response";
            btn.style.color = "{fg}";
          }}, 1800);
        }});
      }}
    </script>
    """
    components.html(html, height=34)


# ---------------------------------------------------------------------------
# Session state management
# ---------------------------------------------------------------------------


def load_conversation_into_state(conv_id: str, history: ChatHistoryManager) -> None:
    """Load messages and persisted FAISS vector index for a conversation."""
    conv = history.get_conversation(conv_id)
    if not conv:
        return

    st.session_state.active_conv_id = conv_id
    st.session_state.messages = history.get_messages(conv_id)
    st.session_state.selected_model = conv.get("model") or Settings.NVIDIA_MODEL
    st.session_state.documents = history.get_documents(conv_id)

    vector_store = FAISSVectorStore.load(conv_id)
    st.session_state.vector_store = vector_store or FAISSVectorStore()


def ensure_session_state(history: ChatHistoryManager) -> None:
    """Initialize all Streamlit session state variables cleanly."""
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"
    if "response_style" not in st.session_state:
        st.session_state.response_style = "short"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "web_search_mode" not in st.session_state:
        st.session_state.web_search_mode = "Auto"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = Settings.NVIDIA_MODEL
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "processed_upload_keys" not in st.session_state:
        st.session_state.processed_upload_keys = set()

    if "active_conv_id" not in st.session_state:
        existing = history.list_conversations()
        if existing:
            load_conversation_into_state(existing[0]["id"], history)
        else:
            new_conv = history.create_conversation(model=st.session_state.selected_model)
            load_conversation_into_state(new_conv["id"], history)


# ---------------------------------------------------------------------------
# Sidebar rendering
# ---------------------------------------------------------------------------


def render_sidebar(
    history: ChatHistoryManager,
    nvidia_ok: bool,
    nvidia_status_msg: str,
    tavily_ok: bool,
    tavily_status_msg: str,
    available_models: list[ModelConfig],
) -> ModelConfig:
    """Render sidebar controls: New Chat, History, PDF Upload, AI Settings, and Status."""
    with st.sidebar:
        top_col1, top_col2 = st.columns([3, 1])
        with top_col1:
            if st.button("＋ New Chat", use_container_width=True, type="primary"):
                new_conv = history.create_conversation(model=st.session_state.selected_model)
                load_conversation_into_state(new_conv["id"], history)
                st.rerun()
        with top_col2:
            theme_icon = "☀️" if st.session_state.theme == "Dark" else "🌙"
            if st.button(theme_icon, use_container_width=True, help="Toggle Dark / Light theme"):
                st.session_state.theme = "Light" if st.session_state.theme == "Dark" else "Dark"
                st.rerun()

        # ---- API Connection Status ----
        st.markdown('<div class="sidebar-section-label">API Status</div>', unsafe_allow_html=True)
        nv_icon = "🟢" if nvidia_ok else "🔴"
        tv_icon = "🟢" if tavily_ok else "🟡"
        st.caption(f"{nv_icon} **NVIDIA NIM:** {nvidia_status_msg}")
        st.caption(f"{tv_icon} **Tavily Search:** {tavily_status_msg}")
        if st.button("Refresh API Status", use_container_width=True):
            Settings.reload()
            cached_nvidia_probe.clear()
            cached_tavily_probe.clear()
            st.rerun()

        # ---- AI Response Settings ----
        st.markdown('<div class="sidebar-section-label">AI Settings</div>', unsafe_allow_html=True)
        model_ids = [m.model_id for m in available_models]
        model_map = {m.model_id: m for m in available_models}

        if st.session_state.selected_model not in model_map and model_ids:
            st.session_state.selected_model = model_ids[0]

        current_idx = model_ids.index(st.session_state.selected_model) if model_ids else 0
        chosen_model_id = st.selectbox(
            "NVIDIA Model",
            options=model_ids,
            index=current_idx,
            format_func=lambda mid: model_map[mid].display_name,
            help="Models supported by your configured NVIDIA NIM endpoint.",
        )
        if chosen_model_id != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model_id
            history.update_model(st.session_state.active_conv_id, chosen_model_id)

        selected_cfg = model_map.get(chosen_model_id) or build_model_config(chosen_model_id)
        st.caption(selected_cfg.description)

        # Only show temperature when supported by the selected NVIDIA endpoint
        if selected_cfg.supports_temperature:
            st.session_state.temperature = st.slider(
                "Temperature (Creativity)",
                min_value=float(selected_cfg.min_temperature),
                max_value=float(selected_cfg.max_temperature),
                value=float(
                    max(
                        selected_cfg.min_temperature,
                        min(selected_cfg.max_temperature, st.session_state.temperature),
                    )
                ),
                step=0.05,
                help="Lower values give focused, factual answers; higher values increase creativity.",
            )
        else:
            st.info("This reasoning endpoint uses a fixed model-controlled temperature.")

        style_label = st.radio(
            "Answer Length",
            options=("Short (Default)", "Detailed"),
            index=0 if st.session_state.response_style == "short" else 1,
            horizontal=True,
        )
        st.session_state.response_style = "short" if style_label.startswith("Short") else "detailed"

        web_options = ("Auto", "On", "Off") if tavily_ok else ("Off",)
        if st.session_state.web_search_mode not in web_options:
            st.session_state.web_search_mode = web_options[0]
        st.session_state.web_search_mode = st.selectbox(
            "Web Search (Tavily)",
            options=web_options,
            index=web_options.index(st.session_state.web_search_mode),
            help="Auto searches Tavily for current/news queries; On always searches; Off disables web search.",
        )

        # ---- PDF Knowledge Base ----
        st.markdown('<div class="sidebar-section-label">PDF Document Chat</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            f"Upload PDF (max {Settings.MAX_PDF_SIZE_MB} MB)",
            type=["pdf"],
            accept_multiple_files=False,
            key=f"pdf_uploader_{st.session_state.active_conv_id}",
            help="Upload text-based PDFs to ask questions with page citations.",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            upload_token = f"{st.session_state.active_conv_id}:{uploaded_file.name}:{len(file_bytes)}"
            if upload_token not in st.session_state.processed_upload_keys:
                with st.spinner(f"Extracting & indexing '{uploaded_file.name}'..."):
                    try:
                        extracted = process_uploaded_pdf(
                            filename=uploaded_file.name,
                            file_bytes=file_bytes,
                            conversation_id=st.session_state.active_conv_id,
                        )
                        existing_hashes = {d["file_hash"] for d in st.session_state.documents}
                        if extracted.file_hash in existing_hashes:
                            st.info(f"'{extracted.filename}' is already indexed in this chat.")
                        else:
                            store: FAISSVectorStore = st.session_state.vector_store
                            added = store.add_chunks(extracted.chunks)
                            store.save(st.session_state.active_conv_id)
                            history.save_document(
                                conv_id=st.session_state.active_conv_id,
                                filename=extracted.filename,
                                file_path=extracted.saved_path,
                                file_hash=extracted.file_hash,
                                page_count=extracted.page_count,
                                chunk_count=added,
                            )
                            st.session_state.documents = history.get_documents(
                                st.session_state.active_conv_id
                            )
                            st.success(
                                f"Indexed '{extracted.filename}' ({extracted.page_count} pages, {added} chunks)."
                            )
                        st.session_state.processed_upload_keys.add(upload_token)
                    except ScannedPDFError as exc:
                        st.error(str(exc))
                        st.session_state.processed_upload_keys.add(upload_token)
                    except (PDFProcessingError, EmbeddingError) as exc:
                        st.error(str(exc))
                        st.session_state.processed_upload_keys.add(upload_token)

        if st.session_state.documents:
            for doc in st.session_state.documents:
                safe_doc_name = html.escape(str(doc["filename"]))
                st.markdown(
                    f'<div class="doc-pill">📄 <strong>{safe_doc_name}</strong>'
                    f'<small>{doc["page_count"]} pages • {doc["chunk_count"]} chunks</small></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Remove Attached PDFs", use_container_width=True):
                history.clear_documents(st.session_state.active_conv_id)
                FAISSVectorStore.delete_persisted(st.session_state.active_conv_id)
                st.session_state.vector_store = FAISSVectorStore()
                st.session_state.documents = []
                st.rerun()

        # ---- Conversation Actions & History ----
        st.markdown('<div class="sidebar-section-label">Current Conversation</div>', unsafe_allow_html=True)
        if st.button(
            "🧹 Clear Chat Context",
            use_container_width=True,
            disabled=len(st.session_state.messages) == 0,
            help="Clear messages in the active conversation.",
        ):
            history.clear_conversation(st.session_state.active_conv_id)
            st.session_state.messages = []
            st.rerun()

        st.markdown('<div class="sidebar-section-label">Chat History</div>', unsafe_allow_html=True)
        conversations = history.list_conversations()
        for conv in conversations:
            is_active = conv["id"] == st.session_state.active_conv_id
            col_open, col_del = st.columns([5, 1])
            label = f"{'▶ ' if is_active else ''}{conv['title']}"
            with col_open:
                if st.button(
                    label,
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type="secondary" if not is_active else "primary",
                ):
                    if not is_active:
                        load_conversation_into_state(conv["id"], history)
                        st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{conv['id']}", help="Delete conversation"):
                    FAISSVectorStore.delete_persisted(conv["id"])
                    history.delete_conversation(conv["id"])
                    remaining = history.list_conversations()
                    if is_active:
                        if remaining:
                            load_conversation_into_state(remaining[0]["id"], history)
                        else:
                            fresh = history.create_conversation(model=st.session_state.selected_model)
                            load_conversation_into_state(fresh["id"], history)
                    elif not remaining:
                        fresh = history.create_conversation(model=st.session_state.selected_model)
                        load_conversation_into_state(fresh["id"], history)
                    st.rerun()

    return selected_cfg


# ---------------------------------------------------------------------------
# Main Chat Rendering & Message Handling
# ---------------------------------------------------------------------------


def render_header(selected_cfg: ModelConfig, nvidia_ok: bool, tavily_ok: bool) -> None:
    """Render top status header above the chat area."""
    nv_cls = "ok" if nvidia_ok else "err"
    nv_label = "NVIDIA NIM Connected" if nvidia_ok else "NVIDIA Key Missing/Invalid"
    tv_cls = "ok" if tavily_ok else "warn"
    tv_label = f"Web: {st.session_state.web_search_mode}" if tavily_ok else "Web: Key Missing"
    pdf_count = len(st.session_state.documents)
    pdf_pill = f'<span class="status-pill ok">📄 {pdf_count} PDF{"s" if pdf_count != 1 else ""}</span>' if pdf_count else ""

    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-title">
                <span class="brand-dot"></span>
                <span>NVIDIA NIM AI Assistant</span>
                <span class="status-pill">{selected_cfg.display_name}</span>
            </div>
            <div class="header-badges">
                <span class="status-pill {nv_cls}">● {nv_label}</span>
                <span class="status-pill {tv_cls}">🌐 {tv_label}</span>
                {pdf_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_screen(tavily_ok: bool) -> None:
    """Render welcome hero and clickable example prompt cards."""
    st.markdown(
        """
        <div class="welcome-hero">
            <div class="welcome-badge">POWERED BY NVIDIA NIM • TAVILY • FAISS</div>
            <h1 class="welcome-title">How can I help you today?</h1>
            <p class="welcome-subtitle">
                Ask questions, explore live web information with verified sources, or upload a PDF
                in the sidebar for page-cited document Q&amp;A.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_PROMPTS):
        with cols[idx % 2]:
            disabled = example["web_mode"] == "On" and not tavily_ok
            btn_label = f"✨ {example['title']}\n\n{example['prompt']}"
            if st.button(btn_label, key=f"example_{idx}", use_container_width=True, disabled=disabled):
                if example["web_mode"] == "On" and tavily_ok:
                    st.session_state.web_search_mode = "On"
                st.session_state.pending_prompt = example["prompt"]
                st.rerun()


def render_message_metadata(msg: dict[str, Any], idx: int) -> None:
    """Render timestamp, web/PDF badges, source links, and copy button for a message."""
    meta = msg.get("metadata") or {}
    ts_str = format_timestamp(msg.get("timestamp"))
    badges: list[str] = []
    if meta.get("used_web_search"):
        badges.append('<span class="context-chip">🌐 Web Search Used</span>')
    if meta.get("used_pdf"):
        badges.append('<span class="context-chip">📄 PDF Context</span>')
    if meta.get("model"):
        badges.append(f"<span>{meta['model']}</span>")

    badge_html = "".join(badges)
    st.markdown(
        f"""
        <div class="msg-meta-row">
            <div class="msg-badges">{badge_html}</div>
            <span>{html.escape(ts_str)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    web_sources = meta.get("web_sources") or []
    if web_sources:
        chips = "".join(
            f'<a class="source-chip" href="{html.escape(src["url"], quote=True)}" target="_blank" rel="noopener noreferrer">'
            f'🔗 {html.escape(src["title"][:48])}</a>'
            for src in web_sources
            if src.get("url")
        )
        st.markdown(f'<div class="source-card-list">{chips}</div>', unsafe_allow_html=True)

    pdf_citations = meta.get("pdf_citations") or []
    if pdf_citations:
        pages_html = "".join(
            f'<span class="source-chip">📄 {html.escape(str(c["filename"]))} • Page {int(c["page"])}</span>'
            for c in pdf_citations
        )
        st.markdown(f'<div class="source-card-list">{pages_html}</div>', unsafe_allow_html=True)

    if msg["role"] == "assistant" and msg.get("content"):
        render_copy_button(msg["content"], key_id=msg.get("id") or f"idx_{idx}", theme=st.session_state.theme)


def render_chat_history() -> None:
    """Render all persisted messages in the active conversation."""
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🧑‍💻" if msg["role"] == "user" else "⚡"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            render_message_metadata(msg, idx)


def handle_user_turn(
    user_prompt: str,
    history: ChatHistoryManager,
    selected_cfg: ModelConfig,
    tavily_ok: bool,
) -> None:
    """Persist user prompt, retrieve web/PDF context if applicable, and stream NVIDIA response."""
    cleaned = user_prompt.strip()
    if not cleaned:
        st.warning("Please enter a non-empty message.")
        return

    conv_id = st.session_state.active_conv_id
    user_msg = history.add_message(conv_id=conv_id, role="user", content=cleaned, timestamp=now_iso())
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(cleaned)
        render_message_metadata(user_msg, len(st.session_state.messages) - 1)

    web_context: str | None = None
    web_sources_meta: list[dict[str, str]] = []
    used_web_search = False

    pdf_context: str | None = None
    pdf_citations_meta: list[dict[str, Any]] = []
    used_pdf = False

    with st.chat_message("assistant", avatar="⚡"):
        # 1. Optional Web Search via Tavily
        if tavily_ok and should_use_web_search(cleaned, st.session_state.web_search_mode):
            with st.spinner("Searching the web with Tavily..."):
                try:
                    search_res = search_web(cleaned, max_results=5)
                    if search_res.sources:
                        web_context = search_res.format_for_prompt()
                        web_sources_meta = search_res.to_metadata_list()
                        used_web_search = True
                except WebSearchError as exc:
                    st.warning(f"Web search unavailable ({exc}). Answering without live web results.")

        # 2. Optional PDF Retrieval via FAISS + NVIDIA Embeddings
        vector_store: FAISSVectorStore = st.session_state.vector_store
        if not vector_store.is_empty:
            with st.spinner("Retrieving relevant passages from uploaded PDF..."):
                try:
                    hits = vector_store.search(cleaned, top_k=Settings.PDF_TOP_K)
                    if hits:
                        pdf_context, pdf_citations_meta = vector_store.format_retrieved_context(hits)
                        used_pdf = True
                except EmbeddingError as exc:
                    st.warning(f"PDF retrieval error ({exc}). Answering without PDF context.")

        # 3. Stream answer from NVIDIA NIM
        temperature = (
            st.session_state.temperature if selected_cfg.supports_temperature else None
        )

        try:
            stream = stream_chat_completion(
                messages=st.session_state.messages,
                model_id=selected_cfg.model_id,
                temperature=temperature,
                response_style=st.session_state.response_style,
                web_context=web_context,
                pdf_context=pdf_context,
            )
            with st.spinner("Thinking with NVIDIA NIM..."):
                first_token = next(stream, None)

            if first_token is None:
                st.error("The NVIDIA NIM endpoint returned an empty response. Please try again.")
                return

            assistant_text = st.write_stream(itertools.chain([first_token], stream))
        except NvidiaAPIError as exc:
            st.error(str(exc))
            return

        if not assistant_text or not str(assistant_text).strip():
            st.error("The NVIDIA NIM endpoint returned an empty response. Please try again.")
            return

        assistant_text = str(assistant_text)
        metadata = {
            "model": selected_cfg.display_name,
            "response_style": st.session_state.response_style,
            "used_web_search": used_web_search,
            "web_sources": web_sources_meta,
            "used_pdf": used_pdf,
            "pdf_citations": pdf_citations_meta,
        }
        assistant_msg = history.add_message(
            conv_id=conv_id,
            role="assistant",
            content=assistant_text,
            timestamp=now_iso(),
            metadata=metadata,
        )
        st.session_state.messages.append(assistant_msg)
        st.rerun()


def main() -> None:
    history = get_history_manager()
    ensure_session_state(history)
    inject_theme_css(st.session_state.theme)

    nvidia_ok, nvidia_msg, discovered_ids = cached_nvidia_probe(
        Settings.NVIDIA_API_KEY, Settings.NVIDIA_BASE_URL
    )
    tavily_ok, tavily_msg = cached_tavily_probe(Settings.TAVILY_API_KEY)
    available_models = get_available_chat_models(discovered_ids)

    selected_cfg = render_sidebar(
        history=history,
        nvidia_ok=nvidia_ok,
        nvidia_status_msg=nvidia_msg,
        tavily_ok=tavily_ok,
        tavily_status_msg=tavily_msg,
        available_models=available_models,
    )

    render_header(selected_cfg=selected_cfg, nvidia_ok=nvidia_ok, tavily_ok=tavily_ok)

    # Capture prompt before deciding whether to show welcome screen
    prompt_to_run = st.session_state.pop("pending_prompt", None)
    user_typed = st.chat_input("Message NVIDIA AI Assistant...")
    if user_typed is not None:
        if not user_typed.strip():
            st.warning("Please type a message before sending.")
        else:
            prompt_to_run = user_typed

    if not st.session_state.messages and not prompt_to_run:
        render_welcome_screen(tavily_ok=tavily_ok)
    else:
        render_chat_history()

    if prompt_to_run:
        handle_user_turn(
            user_prompt=prompt_to_run,
            history=history,
            selected_cfg=selected_cfg,
            tavily_ok=tavily_ok,
        )


if __name__ == "__main__":
    main()
