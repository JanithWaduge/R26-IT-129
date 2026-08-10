FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY app ./app
COPY scripts ./scripts
COPY templates ./templates
RUN mkdir -p /app/storage/sign_videos /app/outputs && chown -R appuser:appgroup /app
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips=127.0.0.1"]
