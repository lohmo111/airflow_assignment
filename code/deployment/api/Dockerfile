# Build context is the project root: docker build -f code/deployment/api/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY code/deployment/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# features.py must be importable under its original module name: the pickled
# pipeline references features.DerivedFeatures.
COPY code/models/features.py ./features.py
COPY code/deployment/api/main.py ./main.py

# The model trained in stage 2 is baked into the image.
COPY models/model.pkl models/model_meta.json ./artifacts/

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
