# NyayaAI

A legal document assistant that explains a document in plain language, flags clauses worth a closer look, and answers questions with a citation to the exact clause. It gives information, not legal advice.

## How it works

```
Upload -> extract text (page-tagged) -> clause-aware chunks -> TF-IDF index
                                                |
              Groq summary + risks <------------+------------> question -> retrieve top clauses
              (every item cites chunk ids)                        -> Groq answers from those clauses only
                                                                  -> citation check -> answer + "Clause 4, page 2"
```

- **Grounded by design:** the model only sees retrieved clauses, must cite their ids, and any item citing an id it was not shown is dropped. If nothing relevant is retrieved, the model is not called and the app says the document does not cover it.
- **Clause-aware chunking:** splits on "1.", "Clause 8", "Section 8" so citations read like the document does.
- **Local retrieval:** xAI has no public embeddings API, so retrieval is TF-IDF with light stemming and a small legal synonym list. It runs on the server, so no extra service is involved. `Retriever` can be swapped for an embeddings store.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # add your GROQ_API_KEY
uvicorn nyayaai.main:app --reload
```

Open http://127.0.0.1:8000. Run tests with `pytest` (the model is mocked, so no key is needed).

`GROQ_MODEL` defaults to `llama-3.3-70b-versatile`; change it in `.env` if you prefer a different model from your Groq console (Groq offers a free tier).

## Security and privacy

- API key comes only from the environment; `.env` is git-ignored.
- Uploads are checked by extension and file signature, capped in size and page count, and never written to disk.
- Documents live in server memory (last 20 kept). Contents are not logged.
- Document text is fenced as data in the prompt to resist prompt injection; the UI renders it with `textContent`.
- Strict security headers, and API docs endpoints are disabled.

## Limits

Scanned PDFs need OCR first. Very long documents are truncated for the summary (flagged in the response). Answers can still be wrong, so always read the cited clause.

## More

- **Languages:** choose English, Hindi or Tamil before uploading. Summaries and answers come back in that language, and Hindi or Tamil questions are translated to English for retrieval (documents are expected to be in English).
- **Rate limit:** 20 API requests per minute per IP (`RATE_LIMIT_PER_MIN`). Behind a proxy, configure forwarded client IPs.
- **Docker:** `docker build -t nyayaai . && docker run -p 8000:8000 --env-file .env nyayaai`

## How it meets the judging criteria

| Criterion | What is in this repo |
|---|---|
| Problem statement alignment | Upload, plain-language explanation, risky clauses, and cited Q&A on the uploaded document. Framed as an information tool, not a lawyer. |
| Code quality | Small single-purpose modules, type hints, a swappable `LLM` protocol, `ruff` config, and CI that lints and tests every push. |
| Security | Env-only secrets, file signature and size checks, in-memory documents, prompt-injection fencing, no HTML injection, CSP and other headers, rate limiting, generic error messages. |
| Efficiency | Local retrieval (no embedding calls), results cached by file hash, no model call when nothing relevant is found, only the top clauses sent per question. |
| Testing | Tests cover chunking, retrieval, citation validation, extraction, the Groq client, error mapping, rate limiting and the API. The model is mocked, so no key is needed. |
| Accessibility | Skip link, labelled controls, live regions, announced errors, focus moves to results, descriptive citation buttons, visible focus, responsive layout, `lang` set for Hindi and Tamil output. |
