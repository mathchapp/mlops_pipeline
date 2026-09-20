"""Ingeniería de variables: ejecutar el entrenamiento desde src/model_training_evaluation.py."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

TARGET = "Pago_atiempo"
NUMERIC_RAW = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]
NOMINAL = ["tipo_credito", "tipo_laboral"]
ORDINAL = ["tendencia_ingresos"]
RAW_COLUMNS = ["fecha_prestamo", *NUMERIC_RAW, *NOMINAL, *ORDINAL]
DERIVED = ["ratio_cuota_salario", "ratio_otros_prestamos_salario", "total_creditos_sectores", "mes_prestamo"]
NUMERIC = [*NUMERIC_RAW, *DERIVED]
TRENDS = ["Desconocida", "Decreciente", "Estable", "Creciente"]
# Se excluyen campos que podrían depender del resultado posterior al préstamo.
EXCLUDED = ["puntaje", "saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor"]


class LoanFeatureBuilder(BaseEstimator, TransformerMixin):
    """Limpieza determinista y atributos derivados, sin estadísticas del dataset."""

    def fit(self, X, y=None):
        self._check(X)
        return self

    @staticmethod
    def _check(X):
        missing = sorted(set(RAW_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")

    def transform(self, X):
        self._check(X)
        data = X[RAW_COLUMNS].copy()
        data["fecha_prestamo"] = pd.to_datetime(data["fecha_prestamo"], errors="coerce")
        for col in NUMERIC_RAW:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)
            data[col] = data[col].where(data[col] >= 0)
        data["edad_cliente"] = data["edad_cliente"].where(data["edad_cliente"].between(18, 100))
        data["puntaje_datacredito"] = data["puntaje_datacredito"].where(
            data["puntaje_datacredito"].between(0, 999)
        )
        for col in [*NOMINAL, *ORDINAL]:
            data[col] = data[col].astype("string").str.strip()
            data[col] = data[col].replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA, "None": pd.NA, "-": pd.NA})
        data["tendencia_ingresos"] = data["tendencia_ingresos"].where(
            data["tendencia_ingresos"].isin(TRENDS[1:])
        )
        for col in [*NOMINAL, *ORDINAL]:
            data[col] = data[col].astype(object).where(data[col].notna(), np.nan)
        salary = data["salario_cliente"].replace(0, np.nan)
        data["ratio_cuota_salario"] = data["cuota_pactada"] / salary
        data["ratio_otros_prestamos_salario"] = data["total_otros_prestamos"] / salary
        data["total_creditos_sectores"] = data[
            ["creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal"]
        ].sum(axis=1, min_count=3)
        data["mes_prestamo"] = data["fecha_prestamo"].dt.month.astype(float)
        data[DERIVED] = data[DERIVED].replace([np.inf, -np.inf], np.nan)
        return data[[*NUMERIC, *NOMINAL, *ORDINAL]]


def build_preprocessor():
    """Imputa y codifica DENTRO del pipeline; el ajuste usa solo entrenamiento."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocida")),
        ("encoder", OrdinalEncoder(categories=[TRENDS], handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("nominal", nominal, NOMINAL), ("ordinal", ordinal, ORDINAL)
    ], remainder="drop", sparse_threshold=0)


def load_and_split(data_path: str | Path):
    """Partición 60/20/20 estratificada; 1 indica incumplimiento (Pago_atiempo=0)."""
    df = pd.read_csv(data_path, low_memory=False)
    missing = sorted(set([TARGET, *RAW_COLUMNS]) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    original_y = pd.to_numeric(df[TARGET], errors="coerce")
    if original_y.isna().any() or not original_y.isin([0, 1]).all():
        raise ValueError("Pago_atiempo debe contener exclusivamente 0 y 1, sin nulos")
    X = df[RAW_COLUMNS].copy()
    y = (original_y == 0).astype(int).rename("incumplimiento")
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
