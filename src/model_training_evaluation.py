"""Modelos supervisados, métricas y gráficos del Avance 2.

Ejecutar: python src/model_training_evaluation.py
Los resultados se guardan fuera de la estructura fija del repositorio.
"""

import argparse
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    balanced_accuracy_score, f1_score, fbeta_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from ft_engineering import EXCLUDED, LoanFeatureBuilder, build_preprocessor, load_and_split


def build_model(name):
    """Mismo preprocesamiento para cada clasificador."""
    classifiers = {
        "Regresión logística": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=.06, max_depth=3, random_state=42
        ),
    }
    if name not in classifiers:
        raise ValueError(f"Modelo desconocido: {name}")
    return Pipeline([
        ("features", LoanFeatureBuilder()), ("preprocess", build_preprocessor()),
        ("classifier", classifiers[name]),
    ])


def summarize_classification(name, y_true, probabilities, threshold=.5):
    """Clase positiva 1 = incumplimiento; el umbral viene de validación."""
    predicted = (probabilities >= threshold).astype(int)
    return {
        "modelo": name, "n": len(y_true), "eventos": int(sum(y_true)), "umbral": threshold,
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "f2": fbeta_score(y_true, predicted, beta=2, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "accuracy": accuracy_score(y_true, predicted),
    }


def choose_threshold_on_validation(y_true, probabilities):
    """Maximiza F2 en validación para priorizar el recall de incumplimientos."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return .5
    p, r = precision[:-1], recall[:-1]
    f2 = np.divide(5 * p * r, 4 * p + r, out=np.zeros_like(p), where=(4 * p + r) > 0)
    return float(thresholds[int(np.argmax(f2))])


def save_charts(summary, probs, y_val, out):
    columns = ["pr_auc", "roc_auc", "recall", "precision", "f2"]
    ax = summary.set_index("modelo")[columns].plot.bar(figsize=(11, 6), ylim=(0, 1), rot=0)
    ax.set(title="Modelos en validación (1 = incumplimiento)", ylabel="Valor de la métrica")
    ax.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out / "comparacion_modelos.png", dpi=160)
    plt.close()

    fig, (pr_ax, roc_ax) = plt.subplots(1, 2, figsize=(13, 5))
    pr_ax.axhline(np.mean(y_val), ls="--", color="gray", label="Prevalencia")
    roc_ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Azar")
    for name, p in probs.items():
        precision, recall, _ = precision_recall_curve(y_val, p)
        fpr, tpr, _ = roc_curve(y_val, p)
        pr_ax.plot(recall, precision, label=name)
        roc_ax.plot(fpr, tpr, label=name)
    pr_ax.set(title="Precision–Recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    roc_ax.set(title="ROC", xlabel="Falsos positivos", ylabel="Recall", xlim=(0, 1), ylim=(0, 1))
    pr_ax.legend()
    roc_ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curvas_validacion.png", dpi=160)
    plt.close(fig)


def main(data_path, out):
    out.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(data_path)
    names = ["Regresión logística", "Random Forest", "Gradient Boosting"]
    fitted, probs, rows, thresholds = {}, {}, [], {}
    for name in names:
        print(f"Entrenando {name}...", flush=True)
        model = build_model(name)
        model.fit(X_train, y_train)
        p = model.predict_proba(X_val)[:, 1]
        fitted[name], probs[name] = model, p
        thresholds[name] = choose_threshold_on_validation(y_val, p)
        rows.append(summarize_classification(name, y_val, p, thresholds[name]))
    summary = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False).reset_index(drop=True)
    winner = str(summary.loc[0, "modelo"])
    summary["seleccionado"] = summary["modelo"].eq(winner)
    summary.to_csv(out / "resumen_validacion.csv", index=False)
    save_charts(summary, probs, y_val, out)

    # La prueba reservada se consulta una vez después de elegir por validación.
    model = fitted[winner]
    test_p = model.predict_proba(X_test)[:, 1]
    test = pd.DataFrame([summarize_classification(winner, y_test, test_p, thresholds[winner])])
    test.to_csv(out / "resumen_prueba.csv", index=False)
    ConfusionMatrixDisplay.from_predictions(
        y_test, (test_p >= thresholds[winner]).astype(int), labels=[0, 1],
        display_labels=["A tiempo", "Incumplimiento"], cmap="Blues", values_format="d",
    )
    plt.title(f"Prueba: {winner} (umbral {thresholds[winner]:.3f})")
    plt.tight_layout()
    plt.savefig(out / "matriz_confusion_prueba.png", dpi=160)
    plt.close()
    joblib.dump(model, out / "modelo_elegido.joblib")
    (out / "README_resultados.md").write_text(
        f"# Avance 2: modelos y evaluación\n\n"
        f"- Clase positiva: incumplimiento (`Pago_atiempo=0`).\n"
        f"- Entrenamiento: {len(X_train):,}; validación: {len(X_val):,}; prueba: {len(X_test):,}.\n"
        f"- Selección por PR-AUC (desempate ROC-AUC) en validación: **{winner}**.\n"
        f"- PR-AUC validación: {summary.loc[0, 'pr_auc']:.4f}; prueba: {test.loc[0, 'pr_auc']:.4f}.\n"
        f"- Umbral seleccionado solo en validación maximizando F2: {thresholds[winner]:.4f}.\n"
        f"- Recall en prueba: {test.loc[0, 'recall']:.3f}; precision: {test.loc[0, 'precision']:.3f}.\n"
        f"- F2 prioriza detectar impagos; revisar el umbral con costos de negocio.\n"
        f"- Variables excluidas por posible fuga temporal: {', '.join(EXCLUDED)}.\n"
        f"- Confirmar que las variables restantes existan al otorgar el crédito.\n"
        f"- Es una línea base con partición estratificada; validar por fecha antes de producción.\n"
        f"- El archivo joblib requiere mantener `src/ft_engineering.py` accesible al cargarlo.\n",
        encoding="utf-8",
    )
    print("\nVALIDACIÓN\n", summary.round(4).to_string(index=False))
    print("\nPRUEBA RESERVADA\n", test.round(4).to_string(index=False))
    print(f"\nResultados: {out.resolve()}")


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project / "Base_de_datos.csv")
    parser.add_argument("--output-dir", type=Path, default=project.parent / "resultados_avance2")
    options = parser.parse_args()
    main(options.data, options.output_dir)
