from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import shutil
import os
from pathlib import Path
from pypdf import PdfReader
from app.rag import answer, _call_sarvam
from app.prompts import GENERAL_SYSTEM_PROMPT
from app.hybrid_retriever import hybrid_retriever
from app.settings import settings

app = FastAPI(title="LegalAid RAG", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskIn(BaseModel):
    question: str
    filter_filename: Optional[str] = None
    user_id: Optional[str] = None   # NEW: identifies the user for scoped retrieval


class ChatIn(BaseModel):
    question: str


# ── API Routes (defined BEFORE static mount) ──────────
@app.get("/health")
def health():
    return {"ok": True, "jurisdiction": settings.JURISDICTION}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(default=None),   # NEW: optional user_id form field
):
    """Upload and index a PDF file tagged with scope=user_upload."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    text += txt + "\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="PDF contains no extractable text (scanned/image-based PDF is not supported).",
            )

        # Tag with scope=user_upload and user_id for isolation
        num_chunks = hybrid_retriever.add_document(
            text, file.filename,
            scope="user_upload",
            user_id=user_id,
        )

        return {
            "filename": file.filename,
            "chunks_added": num_chunks,
            "scope": "user_upload",
            "user_id": user_id,
            "status": "success",
            "message": f"Successfully indexed {num_chunks} chunks from {file.filename}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(payload: AskIn):
    """RAG-powered Q&A with scoped retrieval.
    - If user_id is set: hybrid_search (user uploads + law corpus merged by score)
    - Otherwise: searches law corpus only
    """
    try:
        out = answer(
            payload.question,
            filter_filename=payload.filter_filename,
            user_id=payload.user_id,
        )
        disclaimer = (
            "This is general legal information, not legal advice. "
            "Consult a qualified professional for your specific situation."
        )
        return {"disclaimer": disclaimer, **out}

    except Exception as e:
        print(f"[error] Generation failed: {e}")
        try:
            # Fallback retrieval: scoped to law corpus
            docs = hybrid_retriever.hybrid_search(
                payload.question,
                user_id=payload.user_id,
                top_k=5,
            )
            fallback = (
                "⚠️ **AI generation temporarily unavailable.**\n\n"
                "Below are the most relevant sections from the legal corpus:"
            )
            return {"answer": fallback, "citations": docs, "fallback": True}
        except Exception as e2:
            print(f"[error] Retrieval also failed: {e2}")
            raise HTTPException(status_code=500, detail="Service unavailable.")


@app.post("/chat")
def chat_general(payload: ChatIn):
    """General LegalAid assistant — no document upload required.
    Calls Sarvam AI directly with a helpful legal assistant persona.
    """
    messages = [
        {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
        {"role": "user",   "content": payload.question},
    ]
    try:
        reply = _call_sarvam(messages)
        return {"answer": reply, "citations": []}
    except Exception as e:
        print(f"[error] /chat Sarvam call failed: {e}")
        raise HTTPException(status_code=500, detail="AI service temporarily unavailable.")


# ── Static UI (mounted LAST so API routes take priority) ─
if not os.path.exists("ui"):
    os.makedirs("ui")
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")