# NyayaAI

NyayaAI is an AI-powered legal document information assistant. It extracts text from PDF, DOCX, and TXT files, splits the document into clause-like chunks, retrieves relevant clauses with TF-IDF, and asks a Groq-hosted model to summarize or answer questions using only those retrieved clauses.

It provides information, not legal advice.

## Project structure

```text
nyaya-ai/
├── nyayaai/
│   ├── __init__.py
│   └── main.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── api/
│   └── index.py
├── index.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
└── .env.example
```

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements-dev.txt
copy .env.example .env
uvicorn index:app --reload
```

Open `http://127.0.0.1:8000`.

Add your own Groq API key to `.env`. Never commit `.env`.

## Vercel

Vercel currently supports FastAPI with zero configuration. Keep `pyproject.toml` or `requirements.txt` at the project root and deploy the whole repository. The `api/index.py` entrypoint is also included for the `/api` function layout.

Set these Environment Variables in Vercel:

- `GROQ_API_KEY`
- `GROQ_MODEL` (optional; defaults to `openai/gpt-oss-120b`)
- `MAX_UPLOAD_MB` (optional)

Do not upload `.env` to GitHub or Vercel.

## Docker

```bash
docker build -t nyayaai .
docker run --rm -p 8000:8000 --env-file .env nyayaai
```

## Important limitations

- Scanned PDFs require OCR before upload.
- The document is kept in server memory; it is not persisted to a database.
- AI answers can be wrong. Always read the cited clause and consult a qualified legal professional when legal advice is needed.
