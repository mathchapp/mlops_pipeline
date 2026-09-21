"""Avance 3: monitoreo periódico y panel Streamlit.

Terminal: python src/model_monitoring.py
Panel:    streamlit run src/model_monitoring.py

En modo demostración, el mismo CSV histórico se separa por fecha en referencia
(primer 60 %) y período actual (40 % restante). Esto ilustra el monitoreo; NO
prueba desempeño futuro, pues el modelo del Avance 2 se entrenó con parte de él.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from ft_engineering import DERIVED, LoanFeatureBuilder, NOMINAL, NUMERIC, ORDINAL, RAW_COLUMNS

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT.parent / "resultados_avance2" / "modelo_elegido.joblib"
DEFAULT_OUTPUT = PROJECT.parent / "resultados_avance3"
# mes_prestamo sigue en el modelo, pero una referencia que cubre menos de un
# año produciría falsas alarmas estacionales si se evaluara como drift.
MONITOR_NUMERIC = [col for col in NUMERIC if col != "mes_prestamo"] + ["probabilidad_incumplimiento"]
MONITOR_CATEGORICAL = [*NOMINAL, *ORDINAL]
MIN_PERIOD = 80
MAX_SAMPLE = 500


def load_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta el modelo: {path}. Ejecutá antes python src/model_training_evaluation.py"
        )
    # Cargar únicamente modelos joblib creados por vos: joblib puede ejecutar código.
    return joblib.load(path)


def load_threshold() -> float:
    """Lee el umbral validado en el Avance 2; usa 0,5 si no existe."""
    metrics = DEFAULT_MODEL.parent / "resumen_prueba.csv"
    if metrics.is_file():
        table = pd.read_csv(metrics)
        if "umbral" in table and len(table):
            return float(table.iloc[0]["umbral"])
    warnings.warn("No se encontró el umbral de Avance 2; se usará 0,5.", stacklevel=2)
    return 0.5


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"El CSV necesita estas columnas para el modelo: {missing}")
    copy = data.copy()
    copy["fecha_prestamo"] = pd.to_datetime(copy["fecha_prestamo"], errors="coerce")
    if copy["fecha_prestamo"].isna().any():
        raise ValueError("Hay fechas inválidas o faltantes en fecha_prestamo")
    if copy.empty:
        raise ValueError("El CSV no contiene registros")
    return copy


def demo_reference_current(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición por fecha sin solapamiento: primeros 60 %, últimos 40 %."""
    ordered = validate_input(data).sort_values("fecha_prestamo", kind="stable")
    cut = round(len(ordered) * 0.6)
    if not 0 < cut < len(ordered):
        raise ValueError("Se necesitan al menos dos registros")
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def score_records(raw: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """Tabla de entradas y pronósticos; sin modificar ni reentrenar el modelo."""
    data = validate_input(raw)
    features = LoanFeatureBuilder().fit_transform(data[RAW_COLUMNS])
    scored = data.copy()
    for col in [*NUMERIC, *NOMINAL, *ORDINAL]:
        if col in DERIVED:  # Solo sumar nuevas columnas; la fuente queda intacta.
            scored[col] = features[col]
    probabilities = model.predict_proba(data[RAW_COLUMNS])[:, 1]
    scored["probabilidad_incumplimiento"] = probabilities
    scored["pronostico_incumplimiento"] = (probabilities >= threshold).astype(int)
    scored["periodo"] = data["fecha_prestamo"].dt.to_period("M").astype(str)
    return scored


def proportions_numeric(reference: pd.Series, current: pd.Series):
    """Bins comunes definidos SOLO por referencia; incluye nulos como categoría."""
    r = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(current, errors="coerce").to_numpy(dtype=float)
    finite = r[np.isfinite(r)]
    if not len(finite):
        return None
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, 11)))
    edges = np.r_[-np.inf, edges[1:-1], np.inf]
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    # np.histogram ignora NaN; se agrega un casillero de faltantes explícito.
    r_counts = np.r_[np.histogram(r[np.isfinite(r)], bins=edges)[0], (~np.isfinite(r)).sum()]
    c_counts = np.r_[np.histogram(c[np.isfinite(c)], bins=edges)[0], (~np.isfinite(c)).sum()]
    return r_counts / len(r), c_counts / len(c)


def proportions_categorical(reference: pd.Series, current: pd.Series):
    r = reference.astype("string").fillna("(Nulo)")
    c = current.astype("string").fillna("(Nulo)")
    categories = sorted(set(r.unique()) | set(c.unique()))
    return (
        r.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float) / len(r),
        c.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float) / len(c),
        categories,
    )


def psi_js(reference_p: np.ndarray, current_p: np.ndarray):
    """PSI con suavizado, JS divergence con base 2 (0 a 1)."""
    # Evitar log(0) y conservar suma 1 tras suavizado simétrico.
    eps = 1e-6
    r = (reference_p + eps) / (reference_p + eps).sum()
    c = (current_p + eps) / (current_p + eps).sum()
    psi = float(np.sum((c - r) * np.log(c / r)))
    js_divergence = float(jensenshannon(r, c, base=2) ** 2)
    return psi, js_divergence


def measure_variable(reference: pd.Series, current: pd.Series, kind: str) -> dict:
    """Retorna métricas, p-valores y semáforo de una variable."""
    result = {
        "psi": np.nan, "js_divergence": np.nan, "ks_stat": np.nan,
        "ks_p": np.nan, "chi2_p": np.nan, "cramers_v": np.nan,
        "n_ref": len(reference), "n_actual": len(current),
        "nulos_ref_pct": float(reference.isna().mean() * 100),
        "nulos_actual_pct": float(current.isna().mean() * 100),
        "riesgo": "Sin datos", "recomendacion": "Revisar disponibilidad de datos",
    }
    if len(current) < MIN_PERIOD or len(reference) < MIN_PERIOD:
        result["recomendacion"] = f"Muestra pequeña: mínimo {MIN_PERIOD} registros por período"
        return result
    if kind == "numerica":
        pair = proportions_numeric(reference, current)
        if pair is None:
            return result
        ref_p, cur_p = pair
        r = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
        c = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
        if len(r) >= 20 and len(c) >= 20:
            test = ks_2samp(r, c, method="asymp")
            result.update(ks_stat=float(test.statistic), ks_p=float(test.pvalue))
    else:
        ref_p, cur_p, _ = proportions_categorical(reference, current)
        counts = np.vstack([ref_p * len(reference), cur_p * len(current)])
        counts = counts[:, counts.sum(axis=0) > 0]
        if counts.shape[1] >= 2:
            test = chi2_contingency(counts, correction=False)
            result["cramers_v"] = float(np.sqrt(test.statistic / counts.sum()))
            # Frecuencias esperadas escasas invalidan la aproximación usual.
            if (test.expected_freq >= 5).all():
                result["chi2_p"] = float(test.pvalue)

    result["psi"], result["js_divergence"] = psi_js(ref_p, cur_p)
    # Umbrales de trabajo: calibrar con historia y costos de negocio.
    high = result["psi"] >= .25 or result["js_divergence"] >= .20
    medium = result["psi"] >= .10 or result["js_divergence"] >= .10
    if kind == "numerica":
        evidence = result["ks_stat"] >= .10 and result["ks_p"] < .01
    else:
        evidence = result["cramers_v"] >= .10 and result["chi2_p"] < .01
    high = high or bool(evidence)
    missing_shift = abs(result["nulos_ref_pct"] - result["nulos_actual_pct"]) >= 10
    result["riesgo"] = "Rojo" if high else "Amarillo" if medium or missing_shift else "Verde"
    result["recomendacion"] = {
        "Rojo": "Investigar fuente, calidad y fecha; revisar modelo antes de reentrenar",
        "Amarillo": "Vigilar siguiente período y revisar segmentos",
        "Verde": "Continuar monitoreo mensual",
    }[result["riesgo"]]
    return result


def monitor(reference: pd.DataFrame, current: pd.DataFrame, model, threshold: float, max_sample=MAX_SAMPLE):
    """Muestrea cada mes y compara con referencia; produce cuatro tablas."""
    ref_scored = score_records(reference, model, threshold)
    now_scored = score_records(current, model, threshold)
    # Muestreo reproducible, independiente por mes, con techo máximo.
    ref_sample = ref_scored.sample(min(len(ref_scored), 2000), random_state=42)
    chunks = []
    for _, group in now_scored.groupby("periodo", sort=True):
        chunks.append(group.sample(min(len(group), max_sample), random_state=42))
    sampled = pd.concat(chunks, ignore_index=True)
    metric_rows, period_rows = [], []
    for period, part in sampled.groupby("periodo", sort=True):
        for col in MONITOR_NUMERIC:
            item = measure_variable(ref_sample[col], part[col], "numerica")
            metric_rows.append({"periodo": period, "variable": col, "tipo": "numerica", **item})
        for col in MONITOR_CATEGORICAL:
            item = measure_variable(ref_sample[col], part[col], "categorica")
            metric_rows.append({"periodo": period, "variable": col, "tipo": "categorica", **item})
        period_rows.append({
            "periodo": period, "n_total": int((now_scored["periodo"] == period).sum()),
            "n_muestra": len(part),
            "probabilidad_media": float(part["probabilidad_incumplimiento"].mean()),
            "alertas_modelo_pct": float(part["pronostico_incumplimiento"].mean() * 100),
        })
    metrics = pd.DataFrame(metric_rows)
    periods = pd.DataFrame(period_rows)
    counts = metrics.pivot_table(index="periodo", columns="riesgo", values="variable", aggfunc="count", fill_value=0)
    for col in ["Rojo", "Amarillo", "Verde", "Sin datos"]:
        periods[f"variables_{col.lower().replace(' ', '_')}"] = periods["periodo"].map(counts[col] if col in counts else {}).fillna(0).astype(int)
    periods["psi_max"] = periods["periodo"].map(metrics.groupby("periodo")["psi"].max())
    alerts = metrics[metrics["riesgo"].isin(["Rojo", "Amarillo"])].copy()
    return ref_scored, now_scored, metrics, periods, alerts


def save_reports(current_scored, metrics, periods, alerts, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    current_scored.to_csv(output / "registros_con_pronosticos.csv", index=False)
    metrics.to_csv(output / "metricas_drift.csv", index=False)
    periods.to_csv(output / "resumen_periodos.csv", index=False)
    alerts.to_csv(output / "alertas.csv", index=False)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=PROJECT / "Base_de_datos.csv")
    parser.add_argument("--current", type=Path, help="CSV nuevo; omitir para demo histórica")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    model = load_model(args.model)
    base = pd.read_csv(args.reference, low_memory=False)
    if args.current:
        reference, current = validate_input(base), pd.read_csv(args.current, low_memory=False)
        mode = "CSV nuevo"
    else:
        reference, current = demo_reference_current(base)
        mode = "demostración retrospectiva"
    threshold = load_threshold() if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        raise ValueError("El umbral debe estar entre 0 y 1")
    _, scored, metrics, periods, alerts = monitor(reference, current, model, threshold)
    save_reports(scored, metrics, periods, alerts, args.output_dir)
    print(f"Modo: {mode}; referencia: {len(reference):,}; actual: {len(current):,}; umbral: {threshold:.4f}")
    print(periods.to_string(index=False))
    print(f"Alertas rojas: {(alerts['riesgo'] == 'Rojo').sum()}; amarillas: {(alerts['riesgo'] == 'Amarillo').sum()}")
    print(f"Tablas generadas en: {args.output_dir.resolve()}")


def dashboard():
    import streamlit as st

    st.set_page_config(page_title="Monitoreo crediticio", layout="wide")
    st.title("Monitoreo del riesgo crediticio")
    st.caption("Comparación de solicitudes históricas y actuales; clase positiva = incumplimiento")
    st.info("Demostración retrospectiva: el CSV histórico también participó en el entrenamiento del Avance 2. "
            "Las alertas ilustran drift; no demuestran rendimiento sobre clientes futuros.")
    @st.cache_resource
    def cached_model(path):
        return load_model(Path(path))

    model_path = st.sidebar.text_input("Ruta del modelo", str(DEFAULT_MODEL))
    uploaded = st.sidebar.file_uploader("CSV actual (opcional)", type="csv")
    threshold = st.sidebar.number_input("Umbral del modelo", min_value=0.001, max_value=0.999,
                                         value=float(load_threshold()), step=0.01, format="%.4f")
    try:
        model = cached_model(model_path)
        base = pd.read_csv(PROJECT / "Base_de_datos.csv", low_memory=False)
        if uploaded is None:
            ref, cur = demo_reference_current(base)
            st.sidebar.caption("Modo demostración: 60 % histórico, 40 % posterior.")
        else:
            ref = validate_input(base)
            cur = pd.read_csv(uploaded, low_memory=False)
            st.sidebar.caption("Referencia: CSV original. Actual: CSV cargado.")
        ref_scored, cur_scored, metrics, periods, alerts = monitor(ref, cur, model, threshold)
    except (ValueError, FileNotFoundError, OSError) as error:
        st.error(str(error))
        st.stop()

    red = int((metrics["riesgo"] == "Rojo").sum())
    yellow = int((metrics["riesgo"] == "Amarillo").sum())
    a, b, c, d = st.columns(4)
    a.metric("Solicitudes actuales", f"{len(cur):,}")
    b.metric("Períodos", periods["periodo"].nunique())
    c.metric("Alertas rojas", red)
    d.metric("Alertas amarillas", yellow)
    if red:
        st.error("🔴 Cambio importante: investigar variables, calidad y disponibilidad antes de evaluar reentrenamiento.")
    elif yellow:
        st.warning("🟡 Cambio moderado: observar el próximo período y revisar segmentos.")
    else:
        st.success("🟢 No se detectaron alertas con estas reglas y muestras.")
    if (metrics["riesgo"] == "Sin datos").any():
        st.warning("Algunos períodos tienen muestras insuficientes: su falta de alerta no equivale a estabilidad.")

    st.subheader("Evolución mensual")
    st.line_chart(periods.set_index("periodo")[["psi_max"]])
    st.bar_chart(periods.set_index("periodo")[["variables_rojo", "variables_amarillo"]])
    st.dataframe(periods, width="stretch", hide_index=True)

    st.subheader("Drift por variable y período")
    period = st.selectbox("Período a inspeccionar", sorted(metrics["periodo"].unique(), reverse=True))
    selected = metrics[metrics["periodo"] == period].sort_values("psi", ascending=False)
    st.dataframe(selected, width="stretch", hide_index=True)
    variable = st.selectbox("Comparar distribución histórica vs actual", selected["variable"].tolist())
    reference_values = ref_scored[variable]
    current_values = cur_scored[cur_scored["periodo"] == period][variable]
    if variable in MONITOR_CATEGORICAL:
        r, c, categories = proportions_categorical(reference_values, current_values)
        distribution = pd.DataFrame({"Histórica": r, "Actual": c}, index=categories)
        st.bar_chart(distribution)
    else:
        rv = pd.to_numeric(reference_values, errors="coerce").dropna()
        cv = pd.to_numeric(current_values, errors="coerce").dropna()
        if len(rv) and len(cv):
            low = min(rv.quantile(.01), cv.quantile(.01))
            high = max(rv.quantile(.99), cv.quantile(.99))
            if low < high:
                bins = np.linspace(low, high, 21)
                rh, _ = np.histogram(rv.clip(low, high), bins=bins)
                ch, _ = np.histogram(cv.clip(low, high), bins=bins)
                distribution = pd.DataFrame({"Histórica": rh / rh.sum(), "Actual": ch / ch.sum()},
                                             index=np.round((bins[:-1] + bins[1:]) / 2, 2))
                st.line_chart(distribution)
            else:
                st.caption("Valores constantes: consultar PSI, KS y faltantes en la tabla.")
    st.caption("Distribución recortada al percentil 1–99 solo para la visualización; las métricas usan los datos completos.")

    st.subheader("Acciones recomendadas")
    if alerts.empty:
        st.write("Mantener el monitoreo mensual y comprobar que lleguen suficientes registros.")
    else:
        st.dataframe(alerts[["periodo", "variable", "riesgo", "psi", "js_divergence", "recomendacion"]],
                     width="stretch", hide_index=True)
    st.caption("PSI ≥0,10 amarillo y ≥0,25 rojo; JS ≥0,10/0,20; KS o chi-cuadrado solo con "
               "p<0,01 y efecto ≥0,10. Umbrales iniciales, no calibrados para producción.")
    for title, frame in [
        ("Descargar tabla de métricas", metrics),
        ("Descargar resumen temporal", periods),
        ("Descargar registros con pronósticos", cur_scored),
    ]:
        st.download_button(title, frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name=title.lower().replace(" ", "_") + ".csv", mime="text/csv")


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        running_as_streamlit = get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        running_as_streamlit = False
    dashboard() if running_as_streamlit else cli()
