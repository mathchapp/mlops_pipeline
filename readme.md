# Modelo de riesgo crediticio - PI M5

Proyecto integrador de Data Science para analizar créditos históricos y desarrollar un modelo que prediga si un cliente pagará a tiempo.

## Estructura

- `src/Cargar_datos.ipynb`: carga y controles iniciales de la base.
- `src/comprension_eda.ipynb`: análisis exploratorio univariable, bivariable y multivariable.
- `src/ft_engineering.py`: ingeniería de características (avance posterior).
- `src/model_training_evaluation.py`: entrenamiento y evaluación (avance posterior).
- `src/model_deploy.py`: despliegue del modelo (avance posterior).
- `src/model_monitoring.py`: monitoreo y data drift (avance posterior).
- `Base_de_datos.csv`: datos del proyecto.
- `requirements.txt`: dependencias necesarias.

## Ramas

- `developer`: desarrollo activo.
- `certification`: validación de cambios.
- `master`: versión estable.

## Instalación en Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name mlops_pipeline --display-name "Python (mlops_pipeline)"
```

Luego, en VS Code, abrir los notebooks y seleccionar el kernel `Python (mlops_pipeline)`.
