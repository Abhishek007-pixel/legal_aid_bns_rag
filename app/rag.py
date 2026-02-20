from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import re
import traceback
import requests

from app.hybrid_retriever import hybrid_retriever
from app.prompts import SYSTEM_PROMPT, USER_PROMPT
from app.settings import settings


def _call_sarvam(messages: list) -> str:
    """Call Sarvam AI chat completions API and strip reasoning tags."""
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": settings.SARVAM_API_KEY,
    }
    payload = {
        "model": settings.SARVAM_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 900,
        "reasoning_effort": "low",   # keeps think blocks minimal
    }
    resp = requests.post(settings.SARVAM_API_URL, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise Exception(f"Sarvam API error {resp.status_code}: {resp.text}")

    raw = resp.json()["choices"][0]["message"]["content"]

    # Strip <think>...</think> reasoning blocks (Sarvam-m chain-of-thought)
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return clean


def _build_context(docs: List[Dict]) -> tuple[str, List[Dict]]:
    lines, cites = [], []
    for i, d in enumerate(docs, start=1):
        tag = f"[{i}]"
        src = d.get("source", "unknown")
        fname = d.get("filename", Path(src).name)
        lines.append(f"{tag} (Source: {fname})\n{d['text'][:1200]}")
        cites.append({"ref": tag, "title": d.get("title", "Section"), "where": fname})
    return "\n\n---\n\n".join(lines), cites


def answer(question: str, filter_filename: str = None) -> Dict:
    docs = hybrid_retriever.search(question, filename=filter_filename, top_k=settings.TOP_K)

    if not docs:
        return {
            "answer": "I couldn't find relevant content in the uploaded documents. Please upload a relevant legal document first.",
            "citations": [],
        }

    context, cites = _build_context(docs)
    user_content = USER_PROMPT.format(
        jurisdiction=settings.JURISDICTION,
        question=question,
        context=context,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        answer_text = _call_sarvam(messages)
        return {"answer": answer_text, "citations": cites}

    except Exception as e:
        print("[rag.answer] Sarvam API error:", e)
        traceback.print_exc()
        return {
            "answer": "⚠️ AI generation is temporarily unavailable. Please try again in a moment.",
            "citations": cites,
        }