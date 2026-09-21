# Modelo de riesgo crediticio - PI M5 Avance N°1

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

# Avance N°2

# Avance N°3 

## Proceso y principales hallazgos

1. **Datos:** 10.763 solicitudes y 23 columnas originales; 511 registros (~4,75 %) no pagaron a tiempo. Existen nulos, categorías inválidas en `tendencia_ingresos`, edades superiores a 100 y salarios iguales a cero.
2. **Variables:** numéricas con mediana y RobustScaler, categóricas nominales con moda y OneHotEncoder, ordinales con `Desconocida` y OrdinalEncoder. Se calculan ratios entre cuota/deudas y salario. El modelo excluye `puntaje` y saldos cuya disponibilidad antes del préstamo no está confirmada.
3. **Modelado:** partición estratificada 60/20/20 en entrenamiento/validación/prueba. Se compararon Regresión logística, Random Forest y Gradient Boosting. El ganador por PR-AUC de validación fue Gradient Boosting (~0,136); en prueba obtuvo PR-AUC ~0,130 y recall ~0,725 con precisión ~0,077. Detecta muchos impagos, pero genera numerosas falsas alertas. El umbral se eligió en validación, no en prueba.
4. **Monitoreo:** compara una referencia histórica fija con muestras **mensuales** de solicitudes recientes y calcula drift por variable y pronóstico. Emitir un alerta no demuestra que el modelo perdió rendimiento: se investiga el origen del cambio y, cuando se conozca el resultado real, se evalúa su calidad.

## Ejecutar el monitoreo

Primero generar el modelo local del Avance 2 si no existe:

```powershell
python src/model_training_evaluation.py
```

El archivo `..\resultados_avance2\modelo_elegido.joblib` se utiliza para pronosticar; **cargar solo un archivo creado por vos**. La carpeta de resultados se ubica fuera del repositorio para respetar la estructura exigida.

Demostración con datos del proyecto (primer 60 % cronológico como referencia; 40 % posterior como períodos actuales):

```powershell
python src/model_monitoring.py
```

Panel interactivo:

```powershell
python -m streamlit run src/model_monitoring.py
```

Para monitorear un archivo nuevo con el mismo esquema de entradas:

```powershell
python src/model_monitoring.py --current ruta\a\nuevas_solicitudes.csv
```

En el panel usar **CSV actual (opcional)**. El CSV nuevo requiere las columnas predictoras originales, entre ellas `fecha_prestamo`; `Pago_atiempo` es opcional. El historial `Base_de_datos.csv` queda como referencia. No se reentrena automáticamente con el archivo nuevo.

## Informes y semáforo

`..\resultados_avance3` contiene:

- `registros_con_pronosticos.csv`: entradas, probabilidad de incumplimiento, pronóstico binario y período mensual.
- `metricas_drift.csv`: métricas por mes y variable, nulos, tamaño de muestra, riesgo y recomendación.
- `resumen_periodos.csv`: cantidad de solicitudes, tamaño muestreado, probabilidad media y recuento de alertas.
- `alertas.csv`: variables en amarillo o rojo.

Cada mes se toma una muestra reproducible de **hasta 500** registros y se compara con una muestra de referencia de hasta **2.000**. Si hay menos de **80** registros en el mes, el riesgo es **Sin datos**: nunca se interpreta como verde. Los límites son exploratorios:

| Métrica | Qué muestra | Regla inicial |
|---|---|---|
| KS (numérica) | Distancia máxima entre distribuciones continuas | rojo si distancia ≥0,10 **y** p<0,01 |
| PSI | Diferencias en proporciones de intervalos fijados por referencia | amarillo ≥0,10; rojo ≥0,25 |
| Jensen–Shannon | Divergencia de proporciones (base 2, 0–1) | amarillo ≥0,10; rojo ≥0,20 |
| Chi-cuadrado (categórica) | Si cambiaron las frecuencias de categorías | rojo si p<0,01 **y** V de Cramér ≥0,10; requiere frecuencias esperadas suficientes |
| Proporción de nulos | Calidad/disponibilidad del dato | amarillo si cambia ≥10 puntos porcentuales |

Los umbrales son decisiones iniciales para el ejercicio, **no estándares universales**. Se deben calibrar con historia, tamaño de muestra y costos de negocio. PSI puede crecer exageradamente cuando aparecen categorías raras ausentes en la referencia; investigar el volumen antes de actuar. Se monitorea también la distribución de probabilidad pronosticada. El atributo `mes_prestamo` permanece en el modelo, pero no entra al semáforo: una referencia que no cubre todos los meses produciría falsas alertas de estacionalidad.

## Cómo interpretar y actuar

- **Verde:** continuar el seguimiento mensual.
- **Amarillo:** investigar segmentos y observar el período siguiente.
- **Rojo:** revisar fuente, calidad, cambios de producto y fecha; estudiar performance real antes de decidir reentrenar.
- **Sin datos:** esperar una muestra suficiente o agregar períodos justificados.


