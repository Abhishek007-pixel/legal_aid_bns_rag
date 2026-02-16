from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import shutil
from pathlib import Path
from pypdf import PdfReader
from app.rag import answer, hybrid_retriever
from app.settings import settings

app = FastAPI(title="LegalAid RAG (Hybrid)", version="0.3")

# Mount static files (UI)
from fastapi.staticfiles import StaticFiles
import os
if not os.path.exists("ui"):
    os.makedirs("ui")
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity in this demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskIn(BaseModel):
    question: str
    filter_filename: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True, "jurisdiction": settings.JURISDICTION}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and index a PDF file dynamically."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract Text
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    text += txt + "\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
            
        if not text.strip():
             raise HTTPException(status_code=400, detail="PDF contains no extractable text (might be scanned/image-based)")

        # Index
        num_chunks = hybrid_retriever.add_document(text, file.filename)
        
        return {
            "filename": file.filename,
            "chunks_added": num_chunks,
            "status": "success",
            "message": f"Successfully indexed {num_chunks} chunks from {file.filename}"
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/ask")
def ask(payload: AskIn):
    try:
        # Pass the filter to the RAG function
        out = answer(payload.question, filter_filename=payload.filter_filename)
        
        disclaimer = (
            "This is general legal information, not legal advice. "
            "Consult a qualified professional for your situation."
        )
        return {"disclaimer": disclaimer, **out}
        
    except Exception as e:
        print(f"[error] LLM Generation Failed: {e}")
        # Fallback: Return top chunks directly
        try:
            docs = hybrid_retriever.search(payload.question, top_k=5, filename=payload.filter_filename)
            fallback_answer = (
                "⚠️ **System Notice: AI Generation Unavailable**\n\n"
                "The generative model is currently experiencing issues. "
                "Below are the most relevant sections retrieved from your documents:"
            )
            return {
                "answer": fallback_answer,
                "citations": docs,
                "fallback": True,
                "error": str(e)
            }
        except Exception as e2:
             print(f"[error] Retrieval Failed: {e2}")
             raise HTTPException(status_code=500, detail="Service unavailable (Retrieval + Generation failed)")