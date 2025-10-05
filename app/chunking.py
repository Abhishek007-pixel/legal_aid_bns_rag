from __future__ import annotations
import re
from typing import Iterable, Dict, List
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
import trafilatura

HEADING_RE = re.compile(r"^(Chapter|CHAPTER|Section|SECTION|\u00a7|Sec\.|Article)\b.*", re.I)

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _by_tokens(words: List[str], max_tokens=520, overlap=96) -> List[str]:
    chunks: List[str] = []
    step = max_tokens - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + max_tokens]
        if len(chunk) < 30:
            continue
        chunks.append(" ".join(chunk))
    return chunks

def _split_section_first(text: str) -> List[str]:
    # Split on headings; fallback to paragraphs
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections: List[str] = []
    cur: List[str] = []
    for ln in lines:
        if HEADING_RE.match(ln) and cur:
            sections.append(" ".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append(" ".join(cur))
    return sections if sections else [text]

def parse_pdf(path: Path) -> Iterable[Dict]:
    reader = PdfReader(str(path))
    full: List[str] = []
    for _, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        full.append(txt)
    text = _clean("\n".join(full))
    for sec in _split_section_first(text):
        yield {"title": path.stem, "text": sec, "source": str(path), "url": None}

def parse_html(path: Path) -> Iterable[Dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    extracted = trafilatura.extract(raw) or BeautifulSoup(raw, "html.parser").get_text(" ")
    text = _clean(extracted)
    for sec in _split_section_first(text):
        yield {"title": path.stem, "text": sec, "source": str(path), "url": None}

def chunk_section(section_text: str, max_tokens=520, overlap=96) -> List[str]:
    words = section_text.split()
    return _by_tokens(words, max_tokens=max_tokens, overlap=overlap)
