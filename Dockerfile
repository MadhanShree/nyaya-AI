FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nyayaai nyayaai
COPY static static
COPY index.py index.py
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]
