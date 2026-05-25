FROM python:3.11-slim

# System deps: Tesseract, Poppler (pdf2image), LibreOffice not needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
RUN mkdir -p /app/data /app/logs

EXPOSE 8501
ENV PYTHONUNBUFFERED=1
ENV APP_DATA_DIR=/app/data
ENV APP_LOG_DIR=/app/logs
ENV APP_DB_PATH=/app/data/app.sqlite3
ENV APP_ENV=production

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
