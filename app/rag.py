from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import traceback
from openai import OpenAI

# IMPORTS CHANGED HERE:
from app.hybrid_retriever import hybrid_retriever 
from app.prompts import SYSTEM_PROMPT, USER_PROMPT
from app.settings import settings

# ... [Keep your OpenRouter Client setup here] ...

def _build_context(docs: List[Dict]) -> tuple[str, List[Dict]]:
    lines, cites = [], []
    for i, d in enumerate(docs, start=1):
        tag = f"[{i}]"
        # Handle cases where source/url might be missing gracefully
        src = d.get("source", "unknown")
        fname = d.get("filename", Path(src).name)
        
        lines.append(f"{tag} (Source: {fname}) {d['text'][:1500]}")
        cites.append({"ref": tag, "title": d.get("title", "Section"), "where": fname})
    return "\n\n".join(lines), cites

def answer(question: str, filter_filename: str = None) -> Dict:
    # UPDATED: Call the hybrid retriever with the filter
    docs = hybrid_retriever.search(question, filename=filter_filename, top_k=settings.TOP_K)
    
    if not docs:
        return {
            "answer": "I couldn't find sufficient context in the provided documents.",
            "citations": []
        }

    context, cites = _build_context(docs)
    user = USER_PROMPT.format(jurisdiction=settings.JURISDICTION, question=question, context=context)

    try:
        # ... [Keep your existing LLM call logic here] ...
        # (Copy pasting your existing OpenRouter call for brevity)
        resp = _or_client.chat.completions.create(
            model=settings.OR_MODEL,
            # ... existing params ...
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        return {"answer": content, "citations": cites}

    except Exception as e:
        print("[rag.answer] Error:", e)
        traceback.print_exc()
        return {"answer": "Error generating response.", "citations": []}