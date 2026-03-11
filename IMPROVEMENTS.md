# LegalAid — Improvement & Training Roadmap

A progressive guide to making the LegalAid RAG system smarter over time.

---

## Level 0 — Retrieval Quality Fixes ✅ (Completed Mar 2026)

Fixes applied after Banking Regulation Act retrieval failures were identified:

| Fix | Files Changed | Status |
|---|---|---|
| **Clause-aware chunking** — split on `(a)`, `(b)` clauses; prefix parent section heading to each clause chunk | `app/chunking.py`, `scripts/ingest.py` | ✅ Done |
| **Query expansion** — "What is banking?" → also searches "banking means", "definition of banking", "section 5 banking regulation" | `app/hybrid_retriever.py` | ✅ Done |
| **TOP_K increase** — `TOP_K=5→10`, `RERANK_CANDIDATES=20→30`, `INITIAL_K=30→50` | `.env` | ✅ Done |
| **Stricter grounding prompt** — LLM must quote context when answer exists; must scan for "means"/"defined as" before saying "not found" | `app/prompts.py` | ✅ Done |
| **Section metadata** — each chunk now stores `section_number` + `section_heading` fields | `scripts/ingest.py` | ✅ Done |
| **Upload persistence fix** — user PDFs in-memory only, not written to shared index | `app/hybrid_retriever.py` | ✅ Done |

> **Action required:** Re-run `scripts/ingest.py` to rebuild the index with the new chunking + metadata.

---

## Level 1 — Feed Better Documents ✅ (Do This First — Free)

The system quality is directly tied to the documents indexed. Add authoritative Indian law PDFs:

| Document | Source |
|---|---|
| BNS, IPC, CrPC, IT Act 2000 | [indiacode.nic.in](https://indiacode.nic.in) |
| Supreme Court judgements | [sci.gov.in](https://sci.gov.in) |
| Labour Acts, Consumer Protection Act | [legislative.gov.in](https://legislative.gov.in) |
| State-specific laws | Respective state govt portals |

**Action:** Upload PDFs via the LegalAid UI. Each one gets indexed into BM25 automatically.

---

## Level 1.5 — Fix User Upload Session Persistence ✅ (Completed)

**Problem Solved:** Previously, user uploads permanently populated `data/index/meta.jsonl`.
**Fix Applied:**
- Implemented **In-Memory Only** chunks for user-uploaded documents to maintain privacy.
- Isolated using `user_id`.
- Added a `DELETE /upload/{user_id}` API endpoint to wipe documents explicitly.

*(Optional Next Step: Introduce JWT Auth and an isolated vector db per user for long-lived document sessions.)*

---

## Level 2 — Tune Retrieval Config ✅ (Completed)

Configured for high precision:
```env
# Retrieve more candidates before reranking
TOP_K=8
RERANK_CANDIDATES=20

# Enable Hybrid Search (FAISS + BM25 — best retrieval quality)
USE_FAISS=true
USE_EMBEDDINGS=true

# When hybrid is on, balance keyword vs semantic
BM25_WEIGHT=0.5
VEC_WEIGHT=0.5
```

---

## Level 3 — Collect User Feedback ✅ (Completed)

**Implemented Data Collection:**
Added 👍 / 👎 buttons on the UI capturing user intent. Real-time feedback is written to `data/feedback.jsonl`:

```json
{"ts": "2026-03-06T18:14:26", "rating": "up", "query": "What is theft under BNS?", "answer": "...", "user_id": "usr_x", "session_id": null}
```

**Next Step (Level 3.5):** Build an active learning script that surfaces negatively rated responses for manual review, generating few-shot prompt adjustments or fine-tuning datasets from failures.

---

## Level 4 — Fine-tune the Cross-Encoder Reranker

The current reranker (`cross-encoder/ms-marco-TinyBERT-L-2-v2`) is a general-purpose model.
Fine-tuning it on legal Q&A pairs makes it prefer **legally relevant** chunks over superficially matching words.

**Training data format:**
```python
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

# (query, passage, label) — 1 = relevant, 0 = not relevant
train_samples = [
    ("What is theft under BNS?", "Section 303 BNS defines theft as...", 1),
    ("What is theft under BNS?", "Section 12 of Labour Act defines working hours...", 0),
]

model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
model.fit(train_dataloader=..., epochs=3, warmup_steps=100)
model.save("models/legal-reranker-v1")
```

---

## Level 5 — Fine-tune the Embedding Model

Fine-tuning `all-MiniLM-L6-v2` on legal Q&A pairs via Negative Ranking Loss makes the vector space mapped specifically for legal phrasing (e.g., "dishonest taking of property" → "theft").

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

train_examples = [
    InputExample(texts=["What is theft?", "Section 303 BNS: whoever takes property dishonestly..."]),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
model.save("models/legal-embedder-v1")
```

---

## Level 6 — Add Conversation Memory & Multi-turn Context

### Conversation Memory
Enable chat history carry-forward so users can ask follow-ups naturally:
```python
# Pass history down from frontend and append to messages buffer
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *chat_history,           # [{role: "user", content:...}, {role:"assistant", content:...}]
    {"role": "user", "content": user_content},
]
```

### Prompt Guard
Inject safety triggers for conversational context to prevent jailbreaking or irrelevant hallucinated "legal" facts outside the Indian jurisdiction.

---

## Level 7 — Multi-Agent Agentic Reasoning 🆕

Migrate from single-prompt RAG to a multi-agent orchestrated reasoning framework using LangGraph or Semantic Kernel:
- **Planner Agent:** Breaks down a complex user query ("I got fired without notice and my landlord is evicting me").
- **Legal Researcher Agent(s):** Triggers `hybrid_retriever`, scoped dynamically per sub-query (e.g., one search on Labour Law, one on Rent Control Act).
- **Synthesis Agent:** Cross-references findings, drafts response formatting, and adds disclaimers.

---

## Level 8 — Cloud Scalability & Infrastructure 🆕

| Component | Architecture Shift |
|---|---|
| **Vector DB** | Shift from local FAISS to hosted **Pinecone** or **Qdrant** for distributed horizontal scaling. |
| **Document Store** | Shift SQLite/BM25 JSONL stores to distributed **PostgreSQL + pgvector** or **MongoDB Atlas**. |
| **Authentication** | Wire up Supabase Auth or Clerk to handle user namespaces, preventing session collisions and managing tier-limits. |
| **Deployment** | CI/CD pipeline triggering automated index builds on Vercel/Next.js (for the UI wrapper) alongside a Render/AWS ECS backend. |

---

## Recommended Priority Order (Updated)

| Priority | Action | Effort | Impact |
|---|---|---|---|
| 1 | Analyze user feedback loops to identify retrieval misses (`feedback.jsonl`) | Low | 🔥 High |
| 2 | Add conversational memory via message history array | Medium | 🔥 High |
| 3 | Fine-tune cross-encoder reranker with the new dataset | High | 🔥 High |
| 4 | Fine-tune semantic embedding model | High | 📈 Medium |
| 5 | Multi-agent reasoning integration (LangGraph) | High | 🔥 High |
| 6 | Migrate to managed cloud vector databases (Pinecone/Qdrant) | Medium | 📈 High Scalability |
| 7 | Secure the pipeline with JWT Auth & Supabase namespaces | Medium | 🛡️ Essential |

---

## Minimum Data Requirements for Fine-tuning

| Task | Minimum Samples | Recommended |
|---|---|---|
| Reranker fine-tuning | 500 (query, passage, label) triplets | 2,000+ |
| Embedding fine-tuning | 200 (question, passage) pairs | 1,000+ |
| LLM fine-tuning (LoRA) | 1,000 (question, answer) pairs | 5,000+ |

**Goal:** Keep gathering feedback data. Every `/feedback` interaction directly contributes to overcoming these data thresholds.
