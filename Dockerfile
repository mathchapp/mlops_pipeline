FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/model_artifacts/modelo_elegido.joblib \
    THRESHOLD_PATH=/app/model_artifacts/resumen_prueba.csv

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY Base_de_datos.csv ./Base_de_datos.csv
COPY src ./src

# Genera dentro de la imagen el pipeline ganador y su umbral validado.
RUN python src/model_training_evaluation.py --output-dir /app/model_artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
