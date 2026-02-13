# scripts/ingest.py
from __future__ import annotations  # must be first

# --- make app imports work when running as a module/script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import hashlib
from typing import List, Dict

from openai import OpenAI, AuthenticationError, BadRequestError

from app.chunking import parse_pdf, parse_html, chunk_section
from app.settings import settings
print(f"DEBUG CHECK: settings.USE_EMBEDDINGS is set to: {settings.USE_EMBEDDINGS}")
RAW = Path("data/raw")
PROC = Path("data/processed")
INDEX = Path("data/index")

CHUNKS = PROC / "chunks.jsonl"
META = INDEX / "meta.jsonl"
FAISS_FILE = INDEX / "faiss.index"
SECTION_MAP = INDEX / "section_map.json"


def _ensure_dirs() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    if INDEX.exists() and INDEX.is_file():
        INDEX.unlink()  # if someone created a file named "index", remove it
    INDEX.mkdir(parents=True, exist_ok=True)


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _openrouter_client() -> OpenAI:
    return OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
    )


def _embed_texts_openrouter(client: OpenAI, texts: List[str]):
    import numpy as np
    embs: List[List[float]] = []
    B = 128
    total = len(texts)

    extra_headers = {}
    if getattr(settings, "OR_SITE_URL", None):
        extra_headers["HTTP-Referer"] = settings.OR_SITE_URL
    if getattr(settings, "OR_SITE_NAME", None):
        extra_headers["X-Title"] = settings.OR_SITE_NAME

    for i in range(0, total, B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(
            model=settings.EMBED_MODEL,
            input=batch,
            extra_headers=extra_headers or None,
        )
        embs.extend([d.embedding for d in resp.data])
        print(f"  • embedded {min(i + len(batch), total)}/{total}")

    X = np.array(embs, dtype="float32")
    # Normalize with safety check to avoid division by zero
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Replace zero norms with 1.0 to avoid NaN (shouldn't happen with embeddings, but safety first)
    norms = np.where(norms == 0, 1.0, norms)
    X = X / norms
    return X


# ---- section heading detection (used to build section_map) ----
# --- ---

# ---- UPDATED: Robust section heading detection ----
import re

# We use a list of patterns ordered by reliability.
# 1. Explicit "Section 123" (High confidence, search anywhere)
# 2. "123. Title" (Medium confidence, requires start of line or sentence)
_SEC_RXES = [
    # Matches: "Section 123", "Sec 123", "Sec. 123" anywhere in text
    re.compile(r"(?i)\b(?:section|sec\.?)\s+(?P<num>\d+[A-Za-z]?)"),
    
    # Matches: "123. The Title" (Must look like a heading: Number + Dot + Space + Capital Letter)
    re.compile(r"(?m)^\s*(?P<num>\d+[A-Za-z]?)\.\s+[A-Z]"),
]

def _extract_all_section_numbers(text: str) -> List[str]:
    """Finds ALL section numbers mentioned in a chunk, distinct."""
    found = set()
    for rx in _SEC_RXES:
        # We use finditer to catch multiple sections in one chunk
        for m in rx.finditer(text or ""):
            num = m.group("num")
            if num:
                # Clean up (sometimes OCR gives '123..' or '123 ')
                clean_num = num.strip(" .")
                if len(clean_num) < 6: # Ignore huge numbers/noise
                    found.add(clean_num)
    return list(found)

def _build_section_map(records: List[Dict]) -> Dict[str, Dict]:
    """
    Map section number -> {idx, source, snippet}
    If a chunk mentions multiple sections, we index it for ALL of them.
    """
    secmap: Dict[str, Dict] = {}
    print(f"[sections] Building map from {len(records)} chunks...")
    
    count = 0
    for idx, r in enumerate(records):
        # Find all sections in this chunk
        nums = _extract_all_section_numbers(r["text"])
        
        for num in nums:
            # If we already found this section in a previous chunk, 
            # usually the FIRST time we see it is the "definition", 
            # so we keep the existing one. 
            # BUT: If the previous one was just a mention and this one 
            # looks like a header (shorter text?), we might swap.
            # For now, First-Found-Wins is usually best for legal docs.
            if num not in secmap:
                secmap[num] = {
                    "idx": idx,
                    "source": r["source"],
                    "filename": r.get("filename", "unknown"), # Store filename
                    "snippet": " ".join(r["text"].split())[:160]
                }
                count += 1
                
    print(f"[sections] Identified {count} unique sections.")
    return secmap


def main() -> None:
    _ensure_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Parse & chunk only; skip embedding/index build")
    args = parser.parse_args()

    print(f"[ingest] scanning {RAW.resolve()} ...")
    files = sorted([p for p in RAW.rglob("*") if p.is_file()])
    if not files:
        print("[ingest] No files found in data/raw/. Add PDFs/HTML and retry.")
        sys.exit(1)
    print(f"[ingest] found {len(files)} file(s).")

    # Parse -> section-first -> chunk
    records: List[Dict] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = parse_pdf
        elif suffix in {".html", ".htm", ".txt"}:
            loader = parse_html
        else:
            print(f"  - skip (unsupported): {path.name}")
            continue

        print(f"[parse] {path.name}")
        sec_count = 0
        chunk_count = 0
        for sec in loader(path):
            sec_count += 1
            chunks = chunk_section(sec["text"], max_tokens=520, overlap=96)
            for c in chunks:
                records.append({
                    "id": _hash(c),
                    "title": sec["title"],
                    "text": c,
                    "source": sec["source"],
                    "url": sec["url"],
                    "filename": path.name,
                })
            chunk_count += len(chunks)
        print(f"      sections: {sec_count}, chunks: {chunk_count}")

    if not records:
        print("[ingest] Parsed 0 chunks.")
        sys.exit(2)

    print(f"[write] writing {len(records)} chunk records → {CHUNKS}")
    with open(CHUNKS, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- build and write section_map (works for both BM25-only and vector modes) ----
    secmap = _build_section_map(records)
    with open(SECTION_MAP, "w", encoding="utf-8") as f:
        json.dump(secmap, f, ensure_ascii=False, indent=2)
    print(f"[sections] indexed {len(secmap)} section headings → {SECTION_MAP}")

    if args.dry_run:
        print("[dry-run] skipping embeddings/index build. Done.")
        return

    # ----- BM25-only path -----
    if not getattr(settings, "USE_EMBEDDINGS", False):
        print("[bm25-only] USE_EMBEDDINGS=false → skipping FAISS vector index.")
        print(f"[meta] writing metadata → {META}")
        with open(META, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[done] Wrote metadata for {len(records)} chunks → {META}")
        return

    # ----- Vector path (requires embeddings + FAISS) -----
    import numpy as np  # noqa
    try:
        import faiss  # type: ignore
    except Exception:
        print("[error] FAISS import failed. Install faiss-cpu and retry, or set USE_EMBEDDINGS=false.")
        print("        pip install faiss-cpu")
        sys.exit(3)

    client = _openrouter_client()
    try:
        texts = [r["text"] for r in records]
        print(f"[embed] creating embeddings for {len(texts)} chunks using {settings.EMBED_MODEL} ...")
        X = _embed_texts_openrouter(client, texts)
    except AuthenticationError:
        print("[error] OpenRouter authentication failed. Check OPENROUTER_API_KEY.")
        sys.exit(4)
    except BadRequestError as e:
        print(f"[error] Embedding request rejected: {e}")
        sys.exit(5)
    except Exception as e:
        print(f"[error] Embedding failed: {e}")
        sys.exit(6)

    print(f"[index] building FAISS (dim={X.shape[1]}, n={X.shape[0]}) ...")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(FAISS_FILE))

    print(f"[meta] writing metadata → {META}")
    with open(META, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] Indexed {len(records)} chunks → {FAISS_FILE}")


if __name__ == "__main__":
    main()
