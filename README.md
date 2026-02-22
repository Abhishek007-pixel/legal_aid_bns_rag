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

## 🚀 What I'm Currently Working On

### ✅ Completed
| Feature | Description |
|---|---|
| **Hybrid RAG Pipeline** | FAISS (semantic) + BM25 (keyword) search fused via Reciprocal Rank Fusion (RRF) |
| **Cross-Encoder Reranking** | `ms-marco-TinyBERT-L-2-v2` reranker for precision scoring of retrieved chunks |
| **Scoped Retrieval System** | Documents tagged with scope (`global_law`, `supreme_court`, `labour_law`, `state_law`, `user_upload`) — retrieval is scoped per query type |
| **User Document Upload** | Upload PDFs via the UI; chunks are indexed in-memory for session-scoped retrieval |
| **Sarvam AI Integration** | LLM generation via Sarvam's chat completions API with `<think>` tag stripping |
| **Structured Response Format** | AI outputs include: Legal Summary, Applicable Law, Key Points, Evidence, and Answer |
| **Fallback Retrieval** | When LLM is unavailable, returns top retrieved chunks directly |
| **RAG Evaluation Framework** | `evaluate_rag.py` with metrics: answer relevance, faithfulness, citation accuracy, context precision |
| **Docker Support** | Full containerisation via `Dockerfile` + `docker-compose.yml` |
| **Chat Formatting** | Client-side markdown rendering for headings, lists, bold, and inline citations |
| **Jurisdiction Metadata** | Chunks enriched with `act_name`, `scope`, and `jurisdiction` fields |

### 🔄 In Progress
- Session persistence improvements for user-uploaded documents
- Expanding the indexed law corpus (BNSS, BSA, Consumer Protection Act)

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
│  Chat interface + PDF upload    │
│  Markdown rendering + citations │
└─────────────────────────────────┘
```

---

## 📁 Project Structure

```
legal_aid/
├── app/
│   ├── main.py              # FastAPI app — /ask, /upload, /chat, /health
│   ├── rag.py               # RAG orchestration — retrieval + LLM generation
│   ├── hybrid_retriever.py  # FAISS + BM25 + RRF + Cross-Encoder + Scope logic
│   ├── chunking.py          # PDF text splitter
│   ├── prompts.py           # System & user prompts for Sarvam AI
│   ├── metrics.py           # RAG evaluation metrics
│   └── settings.py          # Env-based config (Pydantic settings)
├── scripts/
│   ├── ingest.py            # Indexes law PDFs with scope/jurisdiction metadata
│   ├── eval.py              # Retrieval accuracy evaluation
│   └── debug_search.py      # Manual search testing
├── ui/
│   ├── index.html           # Main chat UI
│   ├── script.js            # Frontend logic (upload, ask, markdown render)
│   └── style.css            # Styling
├── data/
│   ├── raw/                 # Source law PDFs (BNS, IPC, etc.)
│   ├── index/               # FAISS index + BM25 meta.jsonl
│   └── uploads/             # User-uploaded PDFs (session only)
├── evaluate_rag.py          # End-to-end RAG benchmark runner
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

### 6️⃣ Run the app
```bash
python run_app.py
# or
uvicorn app.main:app --reload
```
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

Run the built-in evaluation framework:
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

Results are saved to `rag_metrics_report.json` and `rag_performance_report.html`.

---

## 🔮 Future Implementations

### 🔜 Short-Term (Next Sprint)

| Feature | Description |
|---|---|
| **Session-safe user uploads** | Keep user PDFs in-memory only (not persisted to disk) to prevent cross-user data leakage |
| **Expand law corpus** | Add BNSS, BSA, Consumer Protection Act, IT Act 2000 to the indexed corpus |
| **👍/👎 Feedback system** | Collect user ratings per answer → `data/feedback.jsonl` for training data |
| **DELETE /upload/{user_id}** | API endpoint to clear a user's uploaded documents on session end |

### 🔮 Medium-Term

| Feature | Description |
|---|---|
| **Conversation memory** | Multi-turn chat with context carry-forward (append previous turns to the LLM prompt) |
| **Fine-tune Cross-Encoder** | Train `ms-marco-TinyBERT-L-2-v2` on legal Q&A pairs for higher reranking precision |
| **Fine-tune Embedding Model** | Domain-adapt `all-MiniLM-L6-v2` on Indian legal text for better semantic mapping |
| **User authentication** | JWT-based auth so uploads are namespaced securely per user account |
| **Multilingual support** | Hindi and regional language query support via Sarvam AI's multilingual capabilities |

### 🚀 Long-Term

| Feature | Description |
|---|---|
| **Voice query integration** | Speech-to-text input for accessibility |
| **Document comparison** | Side-by-side comparison of two legal documents or sections |
| **Case outcome prediction** | ML model trained on Supreme Court judgement outcomes |
| **Lawyer referral system** | Connect users to verified advocates for paid consultations |
| **Mobile app** | React Native / Flutter app wrapping the FastAPI backend |
| **Hugging Face / Streamlit Cloud deployment** | One-click public deployment |

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
