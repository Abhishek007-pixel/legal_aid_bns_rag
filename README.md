# ⚖️ LegalAid — BNS RAG Assistant

> An AI-powered legal assistant for Indian law, built on a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline with scoped document search, cross-encoder reranking, and Sarvam AI LLM generation.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Sarvam AI](https://img.shields.io/badge/LLM-Sarvam%20AI-purple)](https://sarvam.ai)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 What Is This?

**LegalAid** is a Retrieval-Augmented Generation (RAG) application that lets users ask legal questions about Indian law and get grounded, cited answers — fast and for free.

It currently supports:
- **Bharatiya Nyaya Sanhita (BNS)** — India's replacement for the IPC
- **Supreme Court judgements**
- **Labour laws and state-specific acts**
- Any **user-uploaded PDF** (case files, agreements, notices, etc.)

Users ask a question → the system retrieves the most relevant legal passages → Sarvam AI generates a structured, cited answer.

---

## 🌟 Features

| Feature | Description |
|---|---|
| **Hybrid RAG Pipeline** | FAISS (semantic) + BM25 (keyword) search fused via Reciprocal Rank Fusion (RRF). |
| **Scoped Retrieval System** | Documents tagged with scopes (`global_law`, `supreme_court`, `labour_law`, `state_law`, `user_upload`) to ensure high-precision filtering based on query intent. |
| **Cross-Encoder Reranking** | `ms-marco-TinyBERT-L-2-v2` reranker for precision scoring of retrieved chunks. |
| **Session-Safe User Uploads** | Upload PDFs via the UI; chunks are indexed in-memory for session-only retrieval. Isolated by `user_id` and wiped via a dedicated DELETE endpoint. |
| **Sarvam AI Integration** | LLM generation via Sarvam's chat completions API with advanced intent detection and `<think>` tag stripping. |
| **Structured Legal Responses** | AI outputs include: Legal Summary, Applicable Law, Key Points, Evidence, and Answer, with accurate markdown and inline citations. |
| **Feedback System** | 👍/👎 feedback collection per answer, saved to `data/feedback.jsonl` for continuous fine-tuning and evaluation. |
| **Fallback Retrieval** | Returns top retrieved legal chunks directly when LLM generation fails or is unavailable. |
| **Automated Evaluation** | End-to-end evaluation pipeline (`evaluate_rag.py`) measuring answer relevance, faithfulness, citation accuracy, and context precision. |
| **Diagnostic Utilities** | `diagnose_and_run.py` automatically detects missing dependencies, index issues, and port conflicts before launching the app. |
| **Docker Support** | Full containerisation via `Dockerfile` + `docker-compose.yml`. |

---

## 🧠 How It Works

1. **Document Ingestion (`ingest.py`)**: Law PDFs are parsed, split into clause-aware chunks, and embedded into a FAISS vector database. Metadata (like act name, section number, and jurisdiction) is saved to a BM25 index.
2. **Query Processing**: The FastAPI backend receives the user's question and optional uploaded documents.
3. **Intent Detection**: The system extracts the legal context (e.g., scoping the query to `labour_law` or checking for general greetings).
4. **Hybrid Search & Fusion**: The `hybrid_retriever` performs parallel semantic (FAISS) and keyword (BM25) searches. Results are merged using Reciprocal Rank Fusion (RRF).
5. **Reranking**: A cross-encoder model reranks the fused results to prioritize the most legally accurate chunks.
6. **LLM Generation**: The top chunks are injected into a strict system prompt. Sarvam AI generates the detailed legal answer based *only* on the provided context.
7. **Client Rendering**: The Vanilla JS frontend renders the markdown, handles citations natively, and allows the user to leave thumbs-up/down feedback.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         FastAPI Backend         │
│  POST /ask  │  POST /upload     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│      Hybrid Retriever           │
│  FAISS (semantic vectors)       │
│  +  BM25 (keyword matching)     │
│  →  RRF Merge                   │
│  →  Cross-Encoder Reranker      │
│  →  Scope Filter (per query)    │
└────────────┬────────────────────┘
             │  top-k enriched chunks
             ▼
┌─────────────────────────────────┐
│       Sarvam AI LLM             │
│  (sarvam-m model)               │
│  Structured legal answer +      │
│  citations                      │
└────────────┬────────────────────┘
             │
             ▼
        JSON Response
  { answer, citations, disclaimer }
             │
             ▼
┌─────────────────────────────────┐
│     Static UI (HTML/JS/CSS)     │
│  Chat interface + feedback      │
│  Markdown rendering + citations │
└─────────────────────────────────┘
```

---

## 📁 Project Structure

```
legal_aid/
├── app/
│   ├── main.py              # FastAPI app — /ask, /upload, /chat, /feedback
│   ├── rag.py               # RAG orchestration — retrieval + LLM generation
│   ├── hybrid_retriever.py  # FAISS + BM25 + RRF + Cross-Encoder + Scope logic
│   ├── chunking.py          # PDF text splitter
│   ├── prompts.py           # System & user prompts for Sarvam AI
│   └── settings.py          # Env-based config (Pydantic settings)
├── scripts/
│   ├── ingest.py            # Indexes law PDFs with scope/jurisdiction metadata
│   └── eval.py              # Retrieval accuracy evaluation
├── ui/
│   ├── index.html           # Main chat UI
│   ├── script.js            # Frontend logic (upload, ask, markdown render)
│   └── style.css            # Styling
├── data/
│   ├── raw/                 # Source law PDFs (BNS, IPC, etc.)
│   ├── index/               # FAISS index + BM25 meta.jsonl
│   └── uploads/             # User-uploaded PDFs (session only)
├── evaluate_rag.py          # End-to-end RAG benchmark runner
├── diagnose_and_run.py      # Pre-flight diagnostic and auto-repair script
├── test_complete_pipeline.py# Pipeline integration tests
├── IMPROVEMENTS.md          # Detailed improvement & training roadmap
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| **Backend Framework** | FastAPI (Python) |
| **LLM** | Sarvam AI (`sarvam-m`) |
| **Semantic Search** | FAISS + `sentence-transformers/all-MiniLM-L6-v2` |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranker** | `cross-encoder/ms-marco-TinyBERT-L-2-v2` |
| **PDF Parsing** | pypdf |
| **Frontend** | Vanilla HTML + CSS + JavaScript |
| **Containerisation** | Docker + Docker Compose |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Abhishek007-pixel/legal_aid_bns_rag.git
cd legal_aid
```

### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment
Create a `.env` file (see `.env.example`):
```env
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_MODEL=sarvam-m
JURISDICTION=India
TOP_K=5
USE_FAISS=true
USE_EMBEDDINGS=true
```

### 5️⃣ Index the law PDFs
```bash
python scripts/ingest.py
```

### 6️⃣ Run the app with Diagnostics
```bash
python diagnose_and_run.py
```
*(Alternatively: `python run_app.py` or `uvicorn app.main:app --reload`)*

Open `http://localhost:8000` in your browser.

### 🐳 Docker (Alternative)
```bash
docker-compose up --build
```

---

## 💬 Example Queries

- `"What does Section 303 of BNS state?"`
- `"Define theft under the Bharatiya Nyaya Sanhita."`
- `"What is the punishment for rape under BNS?"`
- `"What are the labour working hour limits in India?"`
- `"Summarise the uploaded FIR document."` *(after uploading a PDF)*

---

## 📊 RAG Evaluation

Run the built-in end-to-end evaluation framework:
```bash
python evaluate_rag.py
```

Metrics tracked:
| Metric | Description |
|---|---|
| **Answer Relevance** | Does the answer address the question? |
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Citation Accuracy** | Are citations correctly attributed? |
| **Context Precision** | Are retrieved chunks relevant? |

Results are saved into actionable JSON reports and detailed HTML dashboards.

---

## 🔮 Future Implementations

For a deep dive into the algorithmic progression and fine-tuning pipeline, see [IMPROVEMENTS.md](./IMPROVEMENTS.md). Some upcoming major goals:
- **Conversation Memory**: Context carry-forward per-session.
- **Cross-Encoder Fine-Tuning**: Customizing the reranker on legal Q&A pairs for precision bounds.
- **Multilingual Support**: Hindi query support via Sarvam AI.
- **Agentic Multi-Step Reasoning**: Connecting multiple endpoints to synthesize case summaries.

---

## 🎥 Demo

🎬 **[Watch Demo on YouTube](https://youtu.be/m9klEMLh5MU)**

---

## 👤 Author

**Abhishek Kumar**  
📍 India  
🔗 [GitHub](https://github.com/Abhishek007-pixel) | [LinkedIn](https://www.linkedin.com/in/workwithabhi007/)

---

## ⭐ Contribute

If you find this useful:
- Give it a ⭐ on GitHub
- Report bugs or suggest features via [Issues](https://github.com/Abhishek007-pixel/legal_aid_bns_rag/issues)
- Submit improvements via Pull Requests

---

## ⚠️ Disclaimer

This application provides **general legal information** based on indexed Indian law texts.  
It is **not legal advice**. For specific legal situations, always consult a qualified advocate.
