# LegalAid — Improvement & Training Roadmap

A progressive guide to making the LegalAid RAG system smarter over time.

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

## Level 1.5 — Fix User Upload Session Persistence ⚠️ (Security / Privacy)

**Problem:** When a user uploads a PDF, two things get permanently saved to disk:

| What | Where | Issue |
|---|---|---|
| The PDF file | `data/uploads/filename.pdf` | Never deleted — grows forever |
| Indexed chunks | `data/index/meta.jsonl` (appended) | Loaded back into BM25 on every server restart — permanently pollutes the law index |

**Risks:**
- If `user_id` is not provided (current UI default), User A's private document becomes searchable by User B after a restart
- Index grows unboundedly with every upload
- Re-running `scripts/ingest.py` silently wipes all user-uploaded chunks from the index

**Two fix options:**

### Option A — In-Memory Only (Recommended for now)
Do NOT write user chunks to `meta.jsonl`. Keep them in RAM only (`self.meta` list).  
They are lost on server restart — which matches the user expectation of a "session upload."

```python
# In hybrid_retriever.add_document() — remove the file persistence block for user_upload scope:
if scope != "user_upload":
    with open(self.meta_path, "a", ...) as f:
        ...  # only persist law corpus additions
```

### Option B — Separate User Index with TTL (Better long-term)
Write user uploads to a separate `data/index/user_meta.jsonl`.  
Add a `DELETE /upload/{user_id}` API endpoint or a background job that purges entries older than N hours.

**Action:** Implement Option A as a quick fix. Option B when adding user accounts / auth.

---

## Level 2 — Tune Retrieval Config (`.env` changes only)

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

**Why FAISS?** BM25 only matches keywords. FAISS finds *semantically similar* chunks — so paraphrased
or conceptual questions ("what happens if I don't pay rent?") get much better results.

**Prerequisite:** Re-run `scripts/ingest.py` after enabling FAISS to generate the vector index.

---

## Level 3 — Collect User Feedback (Build Training Data)

Add 👍 / 👎 buttons on each AI response. Store feedback to `data/feedback.jsonl`:

```json
{"question": "What is theft under BNS?", "answer": "...", "rating": "good", "ts": "2025-02-20"}
{"question": "Labour hours limit?", "answer": "...", "rating": "bad", "ts": "2025-02-20"}
```

**Why:** After ~200 ratings you have a dataset to:
- Identify where the system consistently fails (bad-rated answers)
- Use good-rated pairs as few-shot examples in the prompt
- Fine-tune the reranker (Level 4)

**To implement:** Add a `/feedback` POST endpoint in `app/main.py` and feedback buttons in `ui/script.js`.

---

## Level 4 — Fine-tune the Cross-Encoder Reranker

The current reranker (`cross-encoder/ms-marco-TinyBERT-L-2-v2`) is a general-purpose model.
Fine-tuning it on legal Q&A pairs makes it prefer **legally relevant** chunks.

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

**Then update `.env`:**
```env
RERANKER_MODEL=models/legal-reranker-v1
```

---

## Level 5 — Fine-tune the Embedding Model

Fine-tuning `all-MiniLM-L6-v2` on legal Q&A pairs makes the vector space understand legal semantics
(e.g., "dishonest taking of property" → maps close to "theft").

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

train_examples = [
    InputExample(texts=["What is theft?", "Section 303 BNS: whoever takes property dishonestly..."]),
    # ... more (question, relevant_passage) pairs
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
model.save("models/legal-embedder-v1")
```

**Then update `.env`:**
```env
EMBED_MODEL=models/legal-embedder-v1
```

---

## Level 6 — Add Conversation Memory & Better LLM

### Conversation Memory
Store chat history per session so users can ask follow-ups:
```python
# In /ask endpoint, accept history
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *chat_history,           # previous turns
    {"role": "user", "content": user_content},
]
```

### Upgrade LLM
- **Sarvam large model** — for complex multi-part legal reasoning
- **GPT-4o** — via OpenRouter for highest accuracy (requires paid API key)
- **Mistral / LLaMA local** — for fully offline deployment (requires GPU)

---

## Recommended Priority Order

| Priority | Action | Effort | Impact |
|---|---|---|---|
| 1 | Index 10–15 core Indian law PDFs | Low | 🔥 High |
| 2 | Enable FAISS hybrid search | Low | 🔥 High |
| 3 | Add 👍/👎 feedback collection | Medium | 📈 Medium |
| 4 | Fine-tune cross-encoder reranker | High | 🔥 High |
| 5 | Fine-tune embedding model | High | 📈 Medium |
| 6 | Add conversation memory | Medium | 📈 Medium |
| 7 | Upgrade LLM model | Low | 🔥 High |

---

## Minimum Data Requirements for Fine-tuning

| Task | Minimum Samples | Recommended |
|---|---|---|
| Reranker fine-tuning | 500 (query, passage, label) triplets | 2,000+ |
| Embedding fine-tuning | 200 (question, passage) pairs | 1,000+ |
| LLM fine-tuning (LoRA) | 1,000 (question, answer) pairs | 5,000+ |

Start collecting feedback now — every user interaction is potential training data.
