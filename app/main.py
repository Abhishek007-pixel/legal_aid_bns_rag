from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import answer
from app.settings import settings

app = FastAPI(title="LegalAid RAG (Gemini)", version="0.1")

class AskIn(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"ok": True, "jurisdiction": settings.JURISDICTION}

@app.post("/ask")
def ask(payload: AskIn):
    out = answer(payload.question)
    disclaimer = (
        "This is general legal information, not legal advice. "
        "Consult a qualified professional for your situation."
    )
    return {"disclaimer": disclaimer, **out}
