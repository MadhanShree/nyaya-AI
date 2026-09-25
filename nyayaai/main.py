"""FastAPI app: upload -> analyze -> ask. Documents are kept in memory only."""

import hashlib
import os
import time
from collections import OrderedDict, defaultdict, deque
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chunk import chunk_pages
from .extract import ExtractionError, extract_pages
from .llm import LLM, GroqClient, LLMError
from .pipeline import LANGUAGES, analyze, answer_question
from .retrieve import Retriever

load_dotenv()
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
MAX_DOCS = 20
STATIC = Path(__file__).resolve().parent.parent / "static"
STORE: "OrderedDict[str, dict]" = OrderedDict()  # content hash -> parsed document

app = FastAPI(title="NyayaAI", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.update(
        {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
        }
    )
    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


@lru_cache
def _client() -> LLM:
    return GroqClient()


def get_llm() -> LLM:
    try:
        return _client()
    except LLMError as e:
        raise HTTPException(503, str(e)) from e


RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
HITS: "defaultdict[str, deque]" = defaultdict(deque)


def rate_limit(request: Request) -> None:
    """Small per-IP limit so one client cannot burn the model quota."""
    hits, now = HITS[request.client.host if request.client else "unknown"], time.monotonic()
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    hits.append(now)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    language: str = Field("en", pattern="^(en|hi|ta)$")


@app.post("/api/documents", dependencies=[Depends(rate_limit)])
def upload(file: UploadFile = File(...), language: str = Form("en"), llm: LLM = Depends(get_llm)):
    if language not in LANGUAGES:
        raise HTTPException(422, "Choose English, Hindi or Tamil.")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File is too large. The limit is {MAX_BYTES // 1024 // 1024} MB.")
    doc_id = hashlib.sha256(data).hexdigest()[:32]
    try:
        if doc_id not in STORE:  # same file again: reuse chunks and any analysis already made
            chunks = chunk_pages(extract_pages(file.filename or "", data))
            STORE[doc_id] = {"chunks": chunks, "retriever": Retriever(chunks), "analyses": {}}
            while len(STORE) > MAX_DOCS:
                STORE.popitem(last=False)
        doc = STORE[doc_id]
        if language not in doc["analyses"]:
            doc["analyses"][language] = analyze(llm, doc["chunks"], language)
    except ExtractionError as e:
        raise HTTPException(422, str(e)) from e
    except LLMError as e:
        raise HTTPException(502, str(e)) from e
    STORE.move_to_end(doc_id)
    return {
        "doc_id": doc_id,
        "filename": os.path.basename(file.filename or "document")[:100],
        "analysis": doc["analyses"][language],
        "chunks": [c.__dict__ for c in doc["chunks"]],
    }


@app.post("/api/documents/{doc_id}/ask", dependencies=[Depends(rate_limit)])
def ask(doc_id: str, body: Ask, llm: LLM = Depends(get_llm)):
    doc = STORE.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found. Upload it again.")
    try:
        return answer_question(llm, doc["retriever"], body.question.strip(), language=body.language)
    except LLMError as e:
        raise HTTPException(502, str(e)) from e


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
