from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.rag import answer
from app.settings import settings

app = FastAPI(title="LegalAid RAG (Hybrid)", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity in this demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskIn(BaseModel):
    question: str
    filter_filename: Optional[str] = None  # <--- NEW FIELD

@app.get("/health")
def health():
    return {"ok": True, "jurisdiction": settings.JURISDICTION}

@app.post("/ask")
def ask(payload: AskIn):
    # Pass the filter to the RAG function
    out = answer(payload.question, filter_filename=payload.filter_filename)
    
    disclaimer = (
        "This is general legal information, not legal advice. "
        "Consult a qualified professional for your situation."
    )
    return {"disclaimer": disclaimer, **out}