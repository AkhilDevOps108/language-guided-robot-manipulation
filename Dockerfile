FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' appuser

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 && \
    pip install -r requirements.txt

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install -e . --no-deps

USER appuser

EXPOSE 8000
ENV GRIPGROUND_MODEL_PATH=/models/best.pt

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "gripground.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
