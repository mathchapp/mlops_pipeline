"""API FastAPI para disponibilizar el modelo de riesgo crediticio.

Ejecución local:
    python -m uvicorn src.model_deploy:app --reload

La clase positiva es incumplimiento=1. El endpoint /predict admite un registro
o varios registros (batch) en JSON, y también un CSV enviado como cuerpo.
"""

from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, Request
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ft_engineering import RAW_COLUMNS  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_DIR.parent / "resultados_avance2" / "modelo_elegido.joblib"
DEFAULT_THRESHOLD_PATH = PROJECT_DIR.parent / "resultados_avance2" / "resumen_prueba.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
THRESHOLD_PATH = Path(os.getenv("THRESHOLD_PATH", str(DEFAULT_THRESHOLD_PATH)))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "5000"))


class LoanRecord(BaseModel):
    """Esquema de las variables disponibles al solicitar el crédito."""

    # Algunas variables categóricas del CSV original están representadas con
    # números; Pydantic las normaliza a texto igual que el pipeline entrenado.
    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)

    fecha_prestamo: str | None
    capital_prestado: float | None
    plazo_meses: float | None
    edad_cliente: float | None
    salario_cliente: float | None
    total_otros_prestamos: float | None
    cuota_pactada: float | None
    puntaje_datacredito: float | None
    cant_creditosvigentes: float | None
    huella_consulta: float | None
    creditos_sectorFinanciero: float | None
    creditos_sectorCooperativo: float | None
    creditos_sectorReal: float | None
    promedio_ingresos_datacredito: float | None
    tipo_credito: str | None
    tipo_laboral: str | None
    tendencia_ingresos: str | None


app = FastAPI(
    title="API de riesgo crediticio",
    version="1.0.0",
    description=(
        "Predice probabilidad de incumplimiento. Es un ejercicio académico: "
        "la respuesta es una alerta para revisión y no una decisión automática."
    ),
)

EXAMPLE_RECORD = {
    "fecha_prestamo": "2026-09-24",
    "capital_prestado": 150000,
    "plazo_meses": 12,
    "edad_cliente": 35,
    "salario_cliente": 900000,
    "total_otros_prestamos": 50000,
    "cuota_pactada": 25000,
    "puntaje_datacredito": 720,
    "cant_creditosvigentes": 2,
    "huella_consulta": 1,
    "creditos_sectorFinanciero": 1,
    "creditos_sectorCooperativo": 0,
    "creditos_sectorReal": 1,
    "promedio_ingresos_datacredito": 850000,
    "tipo_credito": "7",
    "tipo_laboral": "Empleado",
    "tendencia_ingresos": "Estable",
}
LOAN_RECORD_SCHEMA = LoanRecord.model_json_schema()


@lru_cache(maxsize=1)
def load_model():
    """Carga una única vez el pipeline completo serializado con joblib."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecutar primero: python src/model_training_evaluation.py"
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_threshold() -> float:
    """Recupera el umbral validado; utiliza un valor de respaldo."""
    configured = os.getenv("PREDICTION_THRESHOLD")
    if configured is not None:
        return float(configured)
    if THRESHOLD_PATH.is_file():
        table = pd.read_csv(THRESHOLD_PATH)
        if "umbral" in table and not table.empty:
            return float(table.loc[0, "umbral"])
    return 0.0358


def validate_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Valida el esquema y retorna exclusivamente las columnas del modelo."""
    if not records:
        raise HTTPException(status_code=422, detail="El lote no contiene registros.")
    if len(records) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"El lote supera el máximo de {MAX_BATCH_SIZE} registros.",
        )
    validated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            item = LoanRecord.model_validate(record)
            validated.append(item.model_dump())
        except ValidationError as exc:
            errors.append({"registro": index, "errores": exc.errors(include_url=False)})
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return pd.DataFrame(validated, columns=RAW_COLUMNS)


async def request_to_frame(request: Request) -> pd.DataFrame:
    """Convierte JSON o CSV en un DataFrame validado para predicción batch."""
    content_type = request.headers.get("content-type", "").split(";")[0].lower()
    if content_type == "application/json":
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="JSON inválido.") from exc
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        elif isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise HTTPException(
                status_code=422,
                detail="Enviar un objeto, una lista de objetos o {'records': [...]}",
            )
        return validate_records(payload)

    if content_type in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
        try:
            csv_frame = pd.read_csv(BytesIO(await request.body()))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"CSV inválido: {exc}") from exc
        missing = sorted(set(RAW_COLUMNS) - set(csv_frame.columns))
        if missing:
            raise HTTPException(status_code=422, detail={"columnas_faltantes": missing})
        records = csv_frame.replace({np.nan: None}).to_dict(orient="records")
        return validate_records(records)

    raise HTTPException(
        status_code=415,
        detail="Content-Type admitido: application/json o text/csv.",
    )


@app.get("/")
def root():
    return {
        "servicio": "API de riesgo crediticio",
        "documentacion": "/docs",
        "salud": "/health",
        "prediccion": "/predict",
    }


@app.get("/health")
def health():
    """Informa si el artefacto está disponible y puede cargarse."""
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {exc}") from exc
    return {"status": "ok", "modelo_cargado": True, "version_api": app.version}


@app.post(
    "/predict",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            LOAN_RECORD_SCHEMA,
                            {"type": "array", "items": LOAN_RECORD_SCHEMA},
                        ]
                    },
                    "example": [EXAMPLE_RECORD],
                },
                "text/csv": {
                    "schema": {"type": "string", "format": "binary"},
                    "example": "CSV con las columnas requeridas",
                },
            },
        }
    },
)
async def predict(request: Request):
    """Predice uno o más registros enviados como JSON o CSV."""
    frame = await request_to_frame(request)
    try:
        probabilities = load_model().predict_proba(frame)[:, 1]
        threshold = load_threshold()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo predecir: {exc}") from exc

    predictions = (probabilities >= threshold).astype(int)
    results = [
        {
            "registro": index,
            "probabilidad_incumplimiento": round(float(probability), 6),
            "prediccion": int(prediction),
            "etiqueta": "Revisar riesgo" if prediction else "Sin alerta",
        }
        for index, (probability, prediction) in enumerate(zip(probabilities, predictions))
    ]
    return {
        "cantidad": len(results),
        "umbral": round(threshold, 6),
        "clase_positiva": "incumplimiento",
        "resultados": results,
    }
