from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import traceback
from openai import OpenAI
from app.retriever import retrieve
from app.prompts import SYSTEM_PROMPT, USER_PROMPT
from app.settings import settings

# OpenRouter client via OpenAI SDK
_headers = {}
if settings.OR_SITE_URL:
    _headers["HTTP-Referer"] = settings.OR_SITE_URL
if settings.OR_SITE_NAME:
    _headers["X-Title"] = settings.OR_SITE_NAME

_or_client = OpenAI(
    base_url=settings.OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
)

def _build_context(docs: List[Dict]) -> tuple[str, List[Dict]]:
    lines, cites = [], []
    for i, d in enumerate(docs, start=1):
        tag = f"[{i}]"
        where = d.get("url") or f"{Path(d['source']).name}"
        lines.append(f"{tag} {d['text'][:1200]}")
        cites.append({"ref": tag, "title": d["title"], "where": where})
    return "\n\n".join(lines), cites

def _extractive_fallback(question: str, docs: List[Dict]) -> Dict:
    # minimal extractive summary if the LLM call fails
    snippets = []
    for i, d in enumerate(docs, start=1):
        if not d["text"]:
            continue
        t = d["text"].strip().replace("\n", " ")
        snippets.append(f"[{i}] {t[:300]}...")
    if not snippets:
        return {"answer": "No relevant context found.", "citations": []}
    ans = (
        "**Extractive summary (LLM fallback):**\n\n" +
        "\n".join(f"- {s}" for s in snippets[:8]) +
        "\n\nThis is a direct extract from sources due to LLM unavailability."
    )
    _, cites = _build_context(docs)
    return {"answer": ans, "citations": cites}

def answer(question: str) -> Dict:
    docs = retrieve(question)
    if not docs:
        return {
            "answer": (
                "I couldn't find sufficient authoritative context in the index for this question. "
                "Please add more primary sources (official acts, rules) and try again."
            ),
            "citations": []
        }

    context, cites = _build_context(docs)
    user = USER_PROMPT.format(jurisdiction=settings.JURISDICTION, question=question, context=context)

    try:
        resp = _or_client.chat.completions.create(
            model=settings.OR_MODEL,
            extra_headers=_headers or None,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        return {"answer": content, "citations": cites}

    except Exception as e:
        print("[rag.answer] OpenRouter error:", e)
        traceback.print_exc()
        return _extractive_fallback(question, docs)
