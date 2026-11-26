# NVIDIA NIM Intelligent AI Chatbot

A modern, ChatGPT-inspired AI assistant built with **Python**, **Streamlit**, **NVIDIA NIM API**, **Tavily Web Search**, **PyPDF**, **FAISS**, and **SQLite**.

---

## Features

- **Real-Time Streaming AI Chat (NVIDIA NIM):** Connects to `https://integrate.api.nvidia.com/v1` using the OpenAI-compatible SDK with live token streaming, conversation context memory, concise answers by default, and an optional detailed explanation mode.
- **Modern ChatGPT-Style UI:** Responsive layout with collapsible sidebar, Dark & Light themes, message bubbles, timestamps, loading indicators, copy-to-clipboard buttons, welcome screen with example prompts, and live API connection status badges.
- **Live Web Search (Tavily):** Supports `Auto`, `On`, and `Off` web search modes. Displays a **Web Search Used** badge and clickable source links without fabricating URLs.
- **PDF Document Chat (PyPDF + FAISS + NVIDIA Embeddings):** Upload text-based PDFs (with size and header validation), extract and chunk text with page numbers, index embeddings in FAISS, and ask questions with page-level citations. Automatically detects scanned/image-only PDFs and shows a clear unsupported-document message.
- **Persistent Chat History (SQLite):** Conversations, messages, source citations, and attached PDF records are saved in SQLite so you can create, reopen, clear, or delete chats at any time.
- **Dynamic Model & Parameter Controls:** Discovers supported chat models from your NVIDIA NIM endpoint (with curated Nemotron, Llama, and Mistral defaults), and only exposes parameters supported by the selected endpoint.

---

## Project Structure

```text
nvidia-ai-chatbot/
├── app.py                  # Main Streamlit application & UI
├── config.py               # Environment config & NVIDIA model catalog
├── requirements.txt        # Python dependencies
├── .env.example            # Template for API keys and settings
├── .gitignore              # Keeps .env, DB files, and uploads out of git
├── README.md               # Setup and usage documentation
├── database/
│   ├── __init__.py
│   └── db.py               # SQLite schema & CRUD layer
├── modules/
│   ├── __init__.py
│   ├── chat.py             # NVIDIA NIM client, model discovery & streaming
│   ├── web_search.py       # Tavily web search integration
│   ├── pdf_processor.py    # PyPDF validation, extraction & chunking
│   ├── embeddings.py       # NVIDIA NIM embeddings & FAISS vector store
│   └── history.py          # Conversation & document history service
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Validation, formatting & secret redaction
├── assets/
│   └── style.css           # Custom Dark/Light ChatGPT-like styling
├── uploaded_files/         # Persisted uploaded PDFs and FAISS indexes
└── tests/
    └── test_chatbot.py     # Unit & Streamlit AppTest suite
```

---

## Obtaining API Keys

### 1. NVIDIA NIM API Key (`NVIDIA_API_KEY`)
1. Visit the NVIDIA API Catalog at **https://build.nvidia.com/**.
2. Sign in or create a free NVIDIA Developer account.
3. Open any model card (for example, `nvidia/nemotron-3-ultra-550b-a55b`), click **Get API Key**, and generate a key starting with `nvapi-`.

### 2. Tavily API Key (`TAVILY_API_KEY`)
1. Visit **https://app.tavily.com/** and sign up for a free account.
2. Copy your API key starting with `tvly-` from the Tavily dashboard.

---

## Installation & Setup

### 1. Navigate to the project folder
```bash
cd nvidia-ai-chatbot
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env` and add your real API keys:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Edit `.env`:
```env
NVIDIA_API_KEY=nvapi-your-actual-key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b
NVIDIA_EMBED_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2
TAVILY_API_KEY=tvly-your-actual-key
MAX_PDF_SIZE_MB=20
```

---

## Running the Application

Start the chatbot with:
```bash
streamlit run app.py
```

Then open the local URL printed in your terminal (typically `http://localhost:8501`).

---

## Running Tests

```bash
pytest -v
```

---

## Security Notes

- API keys are loaded strictly from `.env` via `python-dotenv` and are never hardcoded.
- `.env`, SQLite databases, and uploaded files are excluded via `.gitignore`.
- Error handlers redact key patterns (`nvapi-...`, `tvly-...`, `Bearer ...`) before displaying messages in the UI.
- Uploaded files are checked for `.pdf` extension, `%PDF-` magic bytes, and `MAX_PDF_SIZE_MB` limits.
