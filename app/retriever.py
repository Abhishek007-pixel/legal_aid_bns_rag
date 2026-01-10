# app/retriever.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from .settings import settings

INDEX_DIR = Path("data/index")
META_FILE = INDEX_DIR / "meta.jsonl"
SECTION_MAP_FILE = INDEX_DIR / "section_map.json"

# Load meta chunks
_meta: List[Dict] = [json.loads(l) for l in open(META_FILE, encoding="utf-8")]
_docs = [m["text"] for m in _meta]

# optional pre-built section map (may be incomplete)
_section_map: Dict[str, Dict] = {}
if SECTION_MAP_FILE.exists():
    try:
        _section_map = json.loads(SECTION_MAP_FILE.read_text(encoding="utf-8"))
    except Exception:
        _section_map = {}

# BM25 index
_bm25 = BM25Okapi([d.split() for d in _docs]) if settings.USE_BM25 else None

# regexes
_Q_SEC_RE = re.compile(r"\bsec(?:tion)?\.?\s+(\d+[A-Za-z]?)\b", re.I)
_CHUNK_SEC_PATTERNS = [
    re.compile(r"(?mi)\bsection\s+(?P<num>\d+[A-Za-z]?)\b"),
    re.compile(r"(?mi)\bsec\.?\s*(?P<num>\d+[A-Za-z]?)\b"),
    re.compile(r"(?m)^\s*(?P<num>\d+[A-Za-z]?)\s*[\.\:\-–—]\s"),
]
_CHUNK_SEC_PATTERNS_ANY = [re.compile(p.pattern, re.I) for p in _CHUNK_SEC_PATTERNS]  # copy

def _bm25_scores(query: str) -> List[Tuple[int, float]]:
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(query.split())
    pairs = list(enumerate(scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[: settings.INITIAL_K]

def _keyword_bump(query: str, idx: int) -> float:
    qlow = (query or "").lower()
    txt = _meta[idx]["text"].lower()
    boosts: List[str] = []
    if "theft" in qlow:
        boosts += ["theft", "dishonest", "dishonestly", "movable property", "without consent", "intends to take", "takes", "whoever", "definition"]
    bump = 0.0
    for k in boosts:
        if k in txt:
            bump += 0.15
    return min(bump, 0.45) if bump else 0.0

def _find_section_in_chunk_text(text: str, target: str) -> bool:
    """Return True if this chunk contains an explicit heading for target."""
    if not target:
        return False
    for rx in _CHUNK_SEC_PATTERNS_ANY:
        m = rx.search(text or "")
        if m:
            num = (m.group("num") or "").strip().lower() if m and m.groupdict().get("num") else None
            # If pattern captured a num use that; else fallback to searching literal
            if num:
                if num == target.lower():
                    return True
            else:
                # fallback literal search of '229.' style
                if re.search(rf"\b{re.escape(target)}\s*[\.\:\-–—]\s", text):
                    return True
    # also check any literal like "229." somewhere
    if re.search(rf"\b{re.escape(target)}\s*[\.\:\-–—]|\b{re.escape(target)}\b", text):
        # require the numeric to be near punctuation/heading to avoid accidental numeric matches
        if re.search(rf"\b{re.escape(target)}\s*[\.\:\-–—]\s", text) or re.search(rf"(?m)^\s*{re.escape(target)}\s*$", text):
            return True
    return False

def _scan_find_section_index(target: str) -> Optional[int]:
    """Scan the whole meta to find the earliest chunk that looks like the section heading."""
    if not target:
        return None
    for i, m in enumerate(_meta):
        if _find_section_in_chunk_text(m["text"], target):
            return i
    return None

def _find_next_section_in_text(text: str) -> Optional[str]:
    """Return the first section number found in text if any."""
    for rx in _CHUNK_SEC_PATTERNS_ANY:
        m = rx.search(text or "")
        if m:
            return (m.group("num") or "").strip() if m.groupdict().get("num") else None
    # fallback: look for '###. ' style at line start
    m2 = re.search(r"(?m)^\s*(\d+[A-Za-z]?)\s*[\.\:\-–—]\s", text or "")
    if m2:
        return m2.group(1)
    return None

def _stitch_from_index(start_idx: int, target_section: str, max_steps: int = 12, max_chars: int = 4000) -> str:
    pieces: List[str] = []
    chars = 0
    steps = 0
    i = start_idx
    started = False
    while 0 <= i < len(_meta) and steps < max_steps and chars < max_chars:
        t = _meta[i]["text"]
        sec_here = _find_next_section_in_text(t)
        if sec_here:
            if not started:
                # first heading must match the target (tolerant compare)
                if sec_here and sec_here.lower() != target_section.lower():
                    # if first chunk contains a different section, can't start here
                    # but sometimes the chunk contains the number inside body; accept if target present literally
                    if not re.search(rf"\b{re.escape(target_section)}\b", t):
                        break
                started = True
            else:
                # encountered a different section -> stop
                if sec_here.lower() != target_section.lower():
                    break
        pieces.append(t.strip())
        chars += len(t)
        steps += 1
        i += 1
    stitched = "\n\n".join(pieces)
    stitched = re.sub(r"\n{2,}", "\n\n", stitched)
    return stitched

def _section_in_query(query: str) -> Optional[str]:
    m = _Q_SEC_RE.search(query or "")
    return m.group(1) if m else None

def retrieve(query: str) -> List[Dict]:
    """
    Robust BM25 retrieval + deterministic section stitching fallback:
      - if user asked 'section N', try section_map first
      - if not found, scan meta for heading that matches N
      - if found, stitch forward and return that as first result
      - then append BM25 core picks (with keyword bumps)
    """
    target_section = _section_in_query(query)
    results: List[Dict] = []
    used = set()

    # 1) Try deterministic section_map
    if target_section:
        if target_section in _section_map:
            start_idx = int(_section_map[target_section]["idx"])
        else:
            start_idx = _scan_find_section_index(target_section)
        if start_idx is not None:
            stitched = _stitch_from_index(start_idx, target_section)
            m0 = _meta[start_idx]
            key = (m0["source"], stitched[:80])
            used.add(key)
            results.append({
                "rank": 1,
                "score": 2.0,
                "text": stitched,
                "title": m0["title"],
                "source": m0["source"],
                "url": m0.get("url"),
            })

    # 2) BM25 core
    base = _bm25.get_scores(query.split()) if _bm25 is not None else []
    core_pairs = []
    if base is not None and len(base) > 0:
        pairs = list(enumerate(base))
        pairs.sort(key=lambda x: x[1], reverse=True)
        core_pairs = pairs[: settings.INITIAL_K]

    if core_pairs:
        scores_only = [s for _, s in core_pairs]
        smin, smax = min(scores_only), max(scores_only)
        normed = []
        for idx, sc in core_pairs:
            norm = 0.0 if smax == smin else (float(sc) - smin)/(smax - smin)
            norm += _keyword_bump(query, idx)
            # if chunk contains the explicit section heading, strongly bump
            if target_section and _find_section_in_chunk_text(_meta[idx]["text"], target_section):
                norm += 0.75
            normed.append((idx, norm))
        normed.sort(key=lambda x: x[1], reverse=True)
        core = normed[: settings.TOP_K]
        # add core and small neighbors
        rank_offset = len(results)
        for j, (idx, sc) in enumerate(core, start=1):
            m = _meta[idx]
            key = (m["source"], m["text"][:80])
            if key in used:
                continue
            used.add(key)
            results.append({
                "rank": rank_offset + j,
                "score": float(sc),
                "text": m["text"],
                "title": m["title"],
                "source": m["source"],
                "url": m.get("url"),
            })
            # neighbors
            for off in range(1, settings.EXPAND_NEIGHBORS + 1):
                for nb in (idx - off, idx + off):
                    if 0 <= nb < len(_meta):
                        mnb = _meta[nb]
                        keynb = (mnb["source"], mnb["text"][:80])
                        if keynb in used:
                            continue
                        used.add(keynb)
                        results.append({
                            "rank": rank_offset + j,
                            "score": float(sc * 0.95),
                            "text": mnb["text"],
                            "title": mnb["title"],
                            "source": mnb["source"],
                            "url": mnb.get("url"),
                        })

    # final filtering/sorting
    results.sort(key=lambda r: r["score"], reverse=True)
    results = [r for r in results if r["score"] >= settings.MIN_SIM_SCORE]
    return results[: settings.TOP_K]


# FAISS FOR RETRIEVER.PY

#import faiss
#import numpy as np
# 1. Load the Index
#index = faiss.read_index("data/index/faiss.index")

#def search_vectors(query: str):
    # 2. Embed the query (using the same model as ingest!)
 #   query_vec = _embed_single_text(query) 
    
    # 3. Search
    # D = Distances (Scores), I = Indices (Which docs?)
  #  D, I = index.search(query_vec, k=5)
    
   # return I[0], D[0] # Return the top chunk IDs and their scores