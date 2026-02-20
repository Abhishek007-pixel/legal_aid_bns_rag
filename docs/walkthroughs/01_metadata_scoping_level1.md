# Walkthrough — Metadata Scoping System (Level 1 Fix)
**Date:** 2026-02-20  
**Conversation:** Level 1 Improvement — Metadata Filtering & Scoped Retrieval

---

## What Was Done

Implemented a complete metadata scoping system to prevent law PDFs and user-uploaded documents from contaminating each other during retrieval. Added 12 new law PDFs from `law_file/` and re-indexed the entire corpus.

---

## 1. PDFs Organised into Scoped Folders

**18 PDF files** now organised under `data/raw/`:

```
data/raw/
  global_law/      (11 files)
    Banking Regulation Act.pdf
    Bharatiya Nyaya Sanhita (BNS 2023).pdf
    Bharatiya Sakshya Adhiniyam (BSA 2023).pdf
    Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf
    BNS_Part1.pdf
    Code on Wages, 2019.pdf
    Consumer Protection Act, 2019.pdf
    IT Act, 2000.pdf  +  IT_act_2000_updated.pdf
    protection_of_women_from_domestic_violence_act,_2005.pdf
    THE TRANSFER OF PROPERTY ACT, 1882.pdf

  supreme_court/   (2 files)
    LANDMARK JUDGMENTS OF THE SUPREME COURT.pdf
    Supreme Court Judgments.pdf

  labour_law/      (3 files)
    Industrial Relations Code, 2020.pdf
    Labour Act.pdf
    Social Security Code, 2020.pdf

  state_law/       (2 files)
    Delhi Rent Control Act.pdf
    Maharashtra Police Act.pdf
```

---

## 2. Files Changed

| File | Change Summary |
|---|---|
| `scripts/ingest.py` | Scope detection from parent folder name, fail-fast `ValueError` for unknown folders, `act_name` + `jurisdiction` metadata per chunk |
| `app/hybrid_retriever.py` | Scoped BM25/FAISS retrieval, `hybrid_search()` with weighted score merge, intent detection (1.5x boost), confidence threshold, retrieval JSON logging, `user_id` isolation in `add_document()` |
| `app/main.py` | `user_id` field in `AskIn` and `/upload` endpoint; `/upload` tags docs with `scope="user_upload"` |
| `app/rag.py` | `answer()` routes through `hybrid_search()` when `user_id` set, law-corpus-only when not |
| `app/prompts.py` | Context boundary rules 7–10 added to `SYSTEM_PROMPT` to prevent cross-act blending and hallucination |

---

## 3. Ingest Results (Verified ✅)

```
Total chunks indexed:   2301
  global_law:           1188 chunks
  supreme_court:         553 chunks
  labour_law:            377 chunks
  state_law:             183 chunks

Metadata on every chunk:
  scope         ✅  2301/2301
  act_name      ✅  2301/2301
  jurisdiction  ✅  2301/2301
```

---

## 4. Retrieval Architecture (After Change)

```
User Query
   │
   ├─── user_id set? ──YES──► hybrid_search()
   │                              ├── user_upload pool  (scoped by user_id)
   │                              ├── law corpus pool   (global_law + supreme_court + labour_law + state_law)
   │                              ├── intent detection  ("my contract" etc → 1.5x user boost)
   │                              ├── weighted score merge + deduplication
   │                              ├── MIN_SIM_SCORE threshold filter
   │                              └── retrieval JSON log
   │
   └─── no user_id ──────────► search(scope_filter=LAW_SCOPES)
```

---

## 5. Key Guardrails Added

| Guardrail | Detail |
|---|---|
| Fail-fast scope validation | Unknown folder in `data/raw/` → `ValueError` at ingest time |
| User isolation | `user_upload` docs only returned to the matching `user_id` |
| Confidence threshold | `MIN_SIM_SCORE` (default 0.15) filters weak matches post-merge |
| Intent detection | Signals: "my contract/document/case/agreement/file/complaint/fir/petition" |
| Context boundary prompt | Rules 7–10 in SYSTEM_PROMPT prevent LLM from blending unrelated Acts |
| Retrieval logging | Every search emits: `{"ts":..., "query":..., "scopes_returned":..., "num_docs":..., "top_score":...}` |

---

## 6. How to Add More PDFs in Future

1. Place PDF in the correct scope folder under `data/raw/`
2. Re-run ingest:
   ```powershell
   .\.venv\Scripts\python.exe scripts/ingest.py
   ```
3. No code changes needed — scope is auto-detected from folder name.

> **If you add a new scope category** (e.g. `tribunal_orders`), add it to `ALLOWED_SCOPES` in `scripts/ingest.py` and `LAW_SCOPES` in `app/hybrid_retriever.py`.
