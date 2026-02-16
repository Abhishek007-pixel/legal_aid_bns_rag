from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import traceback
import google.generativeai as genai

from app.hybrid_retriever import hybrid_retriever 
from app.prompts import SYSTEM_PROMPT, USER_PROMPT
from app.settings import settings

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)
_gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)

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
    # Call the hybrid retriever with the filter
    docs = hybrid_retriever.search(question, filename=filter_filename, top_k=settings.TOP_K)
    
    if not docs:
        return {
            "answer": "I couldn't find sufficient context in the provided documents.",
            "citations": []
        }

    context, cites = _build_context(docs)
    user_prompt = USER_PROMPT.format(jurisdiction=settings.JURISDICTION, question=question, context=context)

    try:
        # Call Gemini API
        # Create a new model instance with system instruction
        model_with_system = genai.GenerativeModel(
            settings.GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT
        )
        
        response = model_with_system.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.7,
            )
        )
        return {"answer": response.text, "citations": cites}

    except Exception as e:
        print("[rag.answer] Error:", e)
        traceback.print_exc()
        return {"answer": "Error generating response.", "citations": []}