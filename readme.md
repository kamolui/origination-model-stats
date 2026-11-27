# Propuesta de Proyecto Final: Detección de Drift en Modelos de Crédito

*Nov 11, 2026 · @Luis Enrique Patiño @Jesus Armando Rufino* 

---

## Guía de Inicio Rápido y Uso del Repositorio

Este proyecto utiliza [`uv`](https://docs.astral.sh/uv/) como gestor ultrarrápido de paquetes y entornos virtuales de Python (Python 3.12+).

### 1. Instalar `uv`

Si aún no cuentas con `uv` en tu sistema:

* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # O mediante Homebrew en macOS:
  brew install uv
  ```
* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Verifica la instalación:
```bash
uv --version
```

### 2. Clonar y configurar el entorno

Para inicializar el entorno virtual e instalar todas las dependencias bloqueadas en `uv.lock`:

```bash
# Clonar el repositorio y entrar a la carpeta
cd origination-model-stats

# Sincronizar e instalar dependencias automáticamente en .venv
uv sync
```

*(Si se inicia un proyecto nuevo desde cero, se ejecuta `uv init`, pero en este repositorio ya está configurado con `pyproject.toml` y `uv.lock`).*

### 3. Estructura de Datos

Los datos limpios y particionados ya se encuentran listos en `src/data/`:
* `src/data/credito_2008_2013_train.csv` (Referencia histórica / Baseline, 92,095 créditos)
* `src/data/credito_2008_2013_test.csv` (Monitoreo OOT 1, 32,125 créditos)
* `src/data/credito_2008_2013_validation.csv` (Monitoreo OOT 2, 10,350 créditos)

*(La rutina de descarga de Lending Club, curado y división cronológica se encuentra documentada en [Curado_Data.ipynb](notebooks/research/Curado_Data.ipynb)).*

### 4. Ejecutar las pruebas unitarias

Verifica que el entorno y las funciones estadísticas de drift (PSI y KS) funcionen correctamente:

```bash
uv run pytest
```

### 5. Ejemplo de uso del módulo de Drift con Polars

Puedes invocar scripts o módulos dentro del entorno gestionado con `uv run`:

```python
import polars as pl
from drift.metrics_computation import compare_datasets_drift

# 1. Cargar datos
train = pl.read_csv("src/data/credito_2008_2013_train.csv")
test = pl.read_csv("src/data/credito_2008_2013_test.csv")
val = pl.read_csv("src/data/credito_2008_2013_validation.csv")

# 2. Comparar la base de Train contra Test y Validation
drift_results = compare_datasets_drift(
    reference=train,
    targets={"test": test, "validation": val},
    columns=["loan_amnt", "int_rate", "dti", "annual_inc"],
    n_bins=10,
)

print(drift_results)
```

### 6. Gestión de Dependencias con `uv`

```bash
# Agregar una nueva dependencia al proyecto
uv add <nombre-paquete>

# Agregar una dependencia de desarrollo (ej. pytest, ruff)
uv add --dev <nombre-paquete>

# Actualizar todas las dependencias
uv lock --upgrade && uv sync
```

---

## Motivación

Los modelos de crédito se entrenan con datos históricos, pero la población de solicitantes y el entorno económico cambian con el tiempo. Ese cambio (*drift*) puede degradar el desempeño del modelo sin que nadie lo note a tiempo.

En la práctica, la industria monitorea el drift con estadísticos como el PSI (*Population Stability Index*) y la prueba de Kolmogorov-Smirnov (KS), y suele usar umbrales fijos, por ejemplo "PSI > 0.25 indica cambio significativo". Estos umbrales son convenciones sin sustento estadístico formal: no incorporan la incertidumbre del estadístico, que depende del tamaño de muestra y del número de bins. Tampoco hay evidencia sencilla de que una alerta de drift realmente anticipe una caída del desempeño.

Este proyecto aplica técnicas de inferencia basadas en remuestreo y modelos probabilísticos para evaluar qué tan confiables son estas señales.

## Pregunta de investigación y objetivos

**Pregunta:** ¿Las señales estadísticas de drift (PSI, KS) predicen la degradación de un modelo de crédito, y qué tan confiables son los umbrales que se usan comúnmente?

### Objetivos específicos

1. Cuantificar la incertidumbre del PSI y del KS con intervalos de confianza y sesgo estimado.
2. Construir umbrales de alerta basados en datos, en lugar de reglas fijas.
3. Medir la tasa de falsas alarmas y el poder de detección de cada estadístico.
4. Evaluar si las alertas de drift anticipan caídas reales del desempeño (AUC) del modelo.

## Datos

Se usará el conjunto público de préstamos de **Lending Club** (Kaggle), que incluye la fecha de emisión de cada préstamo, características del solicitante y del crédito, y el resultado final (pagado o incumplido). Es un repositorio ajeno a Clara, citado como pide el curso, y no se usará ningún dato interno de la empresa.

### Partición temporal y estrategia de monitoreo (Decisión Acordada)

Se generaron 3 particiones temporales estrictas ordenadas cronológicamente por `issue_d`:

1. **`train` (`credito_2008_2013_train.csv`):** 2008-06-01 a 2012-12-01 (92,095 préstamos, ~70%). **Población base de referencia (Expected).**
2. **`test` (`credito_2008_2013_test.csv`):** 2013-01-01 a 2013-04-01 (32,125 préstamos, ~24%). **Primer periodo de monitoreo fuera de tiempo (Actual).**
3. **`validation` (`credito_2008_2013_validation.csv`):** 2013-05-01 (10,350 préstamos, ~8%). **Segundo periodo de monitoreo fuera de tiempo (Actual).**

**Estrategia de comparación directa:**
El cómputo de PSI y KS se realiza evaluando la desviación contra la población base de entrenamiento:
* **`train` vs `test`**: Evalúa el drift en el periodo inmediato posterior al entrenamiento.
* **`train` vs `validation`**: Evalúa el drift acumulado en el periodo subsiguiente.

*Justificación:* El objetivo en riesgo de crédito es detectar cuándo la población solicitante deja de parecerse a los datos con los que el modelo aprendió a discriminar (`train`). Comparar únicamente `test` vs `validation` ocultaría el drift acumulado respecto al modelo de origen (por ejemplo, `int_rate` muestra un PSI de 0.265 frente a `train`, pero solo 0.038 entre `test` y `validation`).

- **Variable respuesta:** incumplimiento (sí/no), definido a partir del estado final del préstamo (`Charged Off` / `Default` = 1, `Fully Paid` = 0).
- **Variables explicativas:** monto, tasa de interés, plazo, ingreso anual, razón deuda-ingreso, antigüedad laboral y propósito del préstamo, entre otras.
- **Filtro:** solo préstamos con resultado final observado, para evitar sesgo por créditos aún vigentes.

## Alcance

El proyecto se limita a **un solo conjunto de datos**, un modelo de incumplimiento (regresión logística) y dos detectores de drift: PSI y KS, con chi-cuadrada como comparación para variables categóricas.

Quedan fuera del alcance: modelos de redes neuronales o de boosting, despliegue o monitoreo en producción, y cualquier dato o sistema interno de Clara. Las conclusiones se interpretan como evidencia sobre este conjunto de datos, no como reglas generales para todo modelo de crédito.

## Metodología

Cada técnica del curso responde a una parte de la pregunta:

- **Bootstrap e intervalos de confianza:** se remuestrea dentro del periodo de referencia (donde no hay drift real) para obtener la distribución nula del PSI y del KS. De ahí salen umbrales empíricos (por ejemplo, el percentil 95) e intervalos de confianza para cada estadístico en cada periodo.
- **Jackknife:** se estima el sesgo y la varianza del PSI, que depende del número de bins y del tamaño de muestra. Se usa *delete-a-group* (se omiten bloques de observaciones) para que el cómputo sea viable con muchos registros. También permite ver qué variables o grupos de observaciones aportan más al drift.
- **Estimador de máxima verosimilitud:** se ajusta por MLE un modelo logístico de incumplimiento en cada ventana y se comparan los parámetros entre ventanas con pruebas de razón de verosimilitud.
- **Probabilidad bayesiana:** se modela la tasa de incumplimiento de cada periodo con un modelo Beta-Binomial y se calcula la probabilidad de que la tasa del periodo $t$ sea mayor que la de referencia.
- **Validación cruzada:** se usa validación cruzada con origen móvil (*rolling-origin*), que respeta el orden temporal, para medir la caída del AUC a lo largo del tiempo y verificar si las alertas de drift la anticipan.
- **Comparación de detectores:** para cada detector se estiman la tasa de falsas alarmas (periodos sin cambio real marcados como drift) y el poder de detección (cambios inducidos de magnitud conocida que sí se detectan), con intervalos de confianza bootstrap.

## Resultados esperados

- Intervalos de confianza para el PSI y el KS por periodo, con una estimación de su sesgo.
- Umbrales de alerta empíricos, comparados con la regla fija de PSI > 0.25.
- Una tabla comparativa de detectores con su tasa de falsas alarmas y su poder de detección.
- La probabilidad bayesiana de cambio en la tasa de incumplimiento por periodo.
- Una conclusión sobre si las alertas de drift anticipan la caída del AUC, y con cuánta anticipación.

## Referencias

- Lending Club, conjunto de datos de préstamos (Kaggle).
- Efron, B. y Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Gama, J. et al. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4).
- Rabanser, S., Günnemann, S. y Lipton, Z. (2019). Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift. *NeurIPS*.

---

## Cronograma de Actividades y Entregables

| # | Día | Tarea | Resp. | Depende de | Entregable | Comentarios |
|---|-----|-------|-------|------------|------------|-------------|
| 1 | Mar 24 | Acordar ventanas (años o trimestres), variables, y qué datos se reservan para los escenarios (paso 4 de la metodología). Crear repo y ambiente compartido. | Ambos | n/a | Decisiones por escrito | - |
| 2 | Mar 24 | Descargar Lending Club, filtrar préstamos con resultado final, definir incumplimiento y crear las ventanas. | Jesús | 1 | Dataset limpio | • Generar 3 archivos nuevos para no correr la limpieza siempre (train, test, val)<br>• Generar variable objetivo y features (numéricas)<br>• Si hay variables categóricas transformar en numéricas: https://www.kdnuggets.com/2021/05/deal-with-categorical-data-machine-learning.html |
| 3 | Mar 24 | Programar funciones de PSI (con número de bins configurable) y KS, probándolas con datos sintéticos. | Luis | 2 | Módulo de drift | **Completado:** Implementado en `src/drift/metrics_computation.py` con Polars. Pruebas sintéticas en `tests/test_drift.py`. |
| 4 | Mié 25 | Estadística descriptiva: observaciones y tasa de incumplimiento por ventana. | Jesús | 2 | Tablas y gráficas | • x1 mes<br>• Gráfica barra histograma<br>• violin plot |
| 5 | Mié 25 | Bootstrap de PSI y KS en la ventana de referencia (validation), con intervalos de confianza. | Luis | 2, 3 | Distribuciones nulas | Usar validation para comparar resultados |
| 6 | Jue 26 | Jackknife del sesgo y la varianza del PSI según tamaño de muestra y bins. | Luis | 2, 3 | Tabla de sesgo y varianza | - |
| 7 | Jue 26 | Modelo Beta-Binomial: P(θ₂ > θ₁ \| datos) entre ventanas. | Jesús | 4 | Probabilidades por ventana | k=tasa incum |
| 8 | Vie 27 | Umbrales empíricos de PSI y KS (cuantiles de la distribución nula). | Luis | 5 | Umbrales finales | - |
| 9 | Vie 27 | Regresiones logísticas por MLE en cada ventana y comparación de coeficientes. | Both | 2 | Tabla de coeficientes | 30% para as dist |
| 10 | Sáb 28 | Generar escenarios sin cambio y con desplazamientos conocidos, separados de los remuestreos de los umbrales. | Jesús | 1, 2 | Escenarios guardados | - |
| 11 | Sáb 28 | Redactar el borrador de Motivación, Datos y Metodología del informe. | Luis | n/a | Borrador parcial | - |
| 12 | Dom 29 | Comparar reglas convencionales vs. empíricas: falsas alarmas y detección, con intervalos de confianza. | Luis | 8, 10 | Tabla comparativa | - |
| 13 | Dom 29 | Modelo logístico base fijo y AUC por ventana. Análisis de correlación | Jesús | 2 | Serie de AUC | - |
| 14 | Lun 30 | Contrastar las alertas de PSI y KS con θ, coeficientes y AUC. | Ambos | 7, 9, 12, 13 | Hallazgos y conclusiones | - |
| 15 | Lun 30 | Redactar resultados: objetivos 1 a 3 (Luis) y 4 a 5 (Jesús). | Ambos | 14 | Informe completo | - |
| 16 | Mar 1 dic | Armar la presentación (20 min), con las secciones de cada quien. | Ambos | 15 | Diapositivas | - |
| 17 | Mié 2 y jue 3 dic | Revisión cruzada, reproducibilidad del código, ensayo de la presentación. | Ambos | 16 | Versión final | - |
| 18 | Vie 4 dic | Entrega por Canvas. | Ambos | 17 | Informe y presentación | - |
