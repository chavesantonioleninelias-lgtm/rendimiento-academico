# Sistema inteligente para analizar el rendimiento académico

Proyecto de Ciencia de Datos con dos módulos de Machine Learning y un panel web interactivo:

| Módulo | Técnica | Objetivo |
|---|---|---|
| Supervisado | Regresión Logística (baseline) vs Random Forest | Predecir si un estudiante **aprueba o desaprueba** y explicar qué factores pesan más |
| No supervisado | K-Means (con `StandardScaler`) | Agrupar estudiantes por **perfil académico**; k elegido con Método del Codo y Coeficiente de Silueta |

El proyecto se entrega en **dos formas de uso**:

1. **Dashboard interactivo** (`app_rendimiento.py`): se abre en el navegador, con paneles, gráficos animados y un simulador. Es la opción recomendada para la demostración.
2. **Pipeline por terminal** (`pipeline_rendimiento_academico.py`): entrena todo por lotes y guarda modelos, reportes CSV y gráficos PNG.

---

## 1. Contenido de la entrega

```
proyecto_rendimiento/
├── app_rendimiento.py                  # Dashboard (Streamlit)
├── pipeline_rendimiento_academico.py   # Pipeline por terminal
├── requirements.txt                    # Dependencias
├── README.md                           # Este documento
│
├── salidas/                            # (se crea al ejecutar el pipeline)
│   ├── modelos/    clasificador_rf.joblib, clasificador_logreg.joblib,
│   │               kmeans_pipeline.joblib, metadata.json
│   ├── reportes/   comparacion_modelos.csv, importancia_variables.csv, ...
│   └── graficos/   01_comparacion_modelos.png ... 05_clusters_pca.png
│
└── modelos_app/                        # (se crea al pulsar "Guardar modelos" en el dashboard)
```

> **Importante:** si los archivos se descargaron con nombres como `app_rendimiento (1).py`, renómbrelos a los nombres de arriba (sin el `(1)`). Evita errores con espacios y paréntesis en la terminal.

---

## 2. Requisitos

| Requisito | Detalle |
|---|---|
| Sistema operativo | Windows 10/11 (instrucciones principales), macOS y Linux (al final de cada paso) |
| Python | **3.10 o superior** (se desarrolló y probó con 3.12; si usa 3.14 y alguna librería no instala, use 3.12 o 3.13) |
| Navegador | Chrome, Edge o Firefox actualizados (las animaciones usan CSS moderno) |
| Internet | Solo para instalar las librerías. Después funciona sin conexión |
| Espacio / RAM | < 500 MB de disco, 2 GB de RAM suficientes |
| Permisos | No requiere administrador |

Para comprobar su versión de Python, abra **PowerShell** y escriba:

```powershell
python --version
```

Si el comando no se reconoce, pruebe con `py --version`, o use la ruta completa, por ejemplo `& 'C:\Program Files\Python314\python.exe' --version`. En ese caso, sustituya `python` por esa ruta completa en todos los comandos de esta guía.

---

## 3. Instalación paso a paso (Windows / PowerShell)

### Paso 1. Preparar la carpeta

Cree una carpeta (por ejemplo `C:\proyecto_rendimiento`), copie allí los cuatro archivos y abra PowerShell dentro de ella:

```powershell
cd C:\proyecto_rendimiento
```

### Paso 2. Crear un entorno virtual (recomendado)

Aísla las librerías del proyecto y evita conflictos con otros programas.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Al activarse, verá `(.venv)` al inicio de la línea de la terminal.

> Si PowerShell muestra *"la ejecución de scripts está deshabilitada en este sistema"*, ejecute esto (solo afecta a esta ventana) y repita la activación:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### Paso 3. Instalar las dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

La instalación puede tardar unos minutos. Para verificar que todo quedó bien:

```powershell
python -c "import streamlit, plotly, sklearn, pandas, numpy, matplotlib, seaborn, joblib; print('Todo OK')"
```

Debe imprimir `Todo OK`.

### Equivalente en macOS / Linux

```bash
cd ~/proyecto_rendimiento
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 4. Ejecutar el dashboard (recomendado)

Con el entorno activado y dentro de la carpeta del proyecto:

```powershell
python -m streamlit run app_rendimiento.py
```

- Se abrirá el navegador en `http://localhost:8501`. Si no se abre solo, copie esa dirección en el navegador.
- Si Streamlit pide un correo electrónico la primera vez, simplemente presione **Enter**.
- La **primera carga tarda entre 10 y 40 segundos** porque entrena los modelos (muestra un indicador "Entrenando modelos…"). Después los resultados quedan en caché y la navegación es inmediata.
- Para detener el servidor: `Ctrl + C` en la terminal.

Opciones útiles:

```powershell
# Otro puerto, si el 8501 está ocupado
python -m streamlit run app_rendimiento.py --server.port 8502

# Desactivar el envío de estadísticas de uso anónimas
python -m streamlit run app_rendimiento.py --browser.gatherUsageStats false
```

> Los datos de los estudiantes **se procesan localmente** en su computadora; el dashboard no los envía a ningún servidor externo.

### 4.1 Guía de uso del dashboard

**Barra lateral (izquierda)**

| Control | Función |
|---|---|
| Fuente de datos | *Datos sintéticos de demostración* (por defecto) o *Subir mi CSV* |
| Número de estudiantes | Solo con datos sintéticos (300–3000) |
| Descargar plantilla CSV | Archivo de ejemplo con el formato exacto requerido |
| Nota mínima aprobatoria | Solo aplica si el CSV no trae la columna `aprobo` (por defecto 11) |
| Número de clusters (k) | *Automático* (máxima silueta) o un valor manual de 2 a 8 |
| Semáforo de riesgo | Umbrales de probabilidad para clasificar riesgo Alto / Medio / Bajo |
| Guardar modelos (.joblib) | Exporta los 3 modelos a la carpeta `modelos_app/` |

**Pestañas**

| Pestaña | Qué muestra | Qué debe observar el instructor |
|---|---|---|
| Exploración | Histogramas por resultado, matriz de correlación, dispersión con ejes elegibles | Qué variables separan a quienes aprueban de quienes no |
| Modelo supervisado | Métricas, curvas ROC, matrices de confusión, importancia de variables, coeficientes, justificación de KNN | Comparación baseline vs Random Forest; qué factores son más importantes |
| Clusters | Codo, silueta, mapa PCA, perfiles, radar, tasa de aprobación por grupo | Si los grupos tienen sentido pedagógico y qué tan separados están |
| Simulador | Deslizadores para armar un estudiante; velocímetro de probabilidad, grupo asignado, radar y recomendaciones | Cómo cambia la predicción al variar asistencia, entregas, notas, etc. |
| Estudiantes en riesgo | Semáforo, distribución por grupo y tabla filtrable descargable en CSV | Lista de alerta temprana priorizada por probabilidad |
| Guía | Resumen de interpretación | Referencia rápida para los estudiantes |

### 4.2 Guion sugerido para una demostración de 15 minutos

1. **(2 min)** Abrir el dashboard con datos sintéticos y explicar los cinco indicadores superiores.
2. **(3 min)** *Exploración*: elegir `asistencia_pct` y mostrar cómo se separan aprobados y desaprobados.
3. **(4 min)** *Modelo supervisado*: comparar Regresión Logística vs Random Forest; mostrar la importancia por permutación y abrir el desplegable "¿Por qué se descarta KNN?".
4. **(3 min)** *Clusters*: mostrar codo y silueta; cambiar k en la barra lateral (por ejemplo 2 → 4) y comentar cómo cambian los perfiles.
5. **(2 min)** *Simulador*: bajar la asistencia y las entregas de un estudiante y ver cómo cae la probabilidad y aparecen recomendaciones.
6. **(1 min)** *Estudiantes en riesgo*: filtrar por riesgo Alto y descargar el CSV.

---

## 5. Ejecutar el pipeline por terminal (opcional)

Útil para reproducir el entrenamiento sin interfaz gráfica o integrarlo en otros procesos.

```powershell
# Datos sintéticos, incluyendo KNN como referencia empírica
python pipeline_rendimiento_academico.py --benchmark-knn

# Con datos reales
python pipeline_rendimiento_academico.py --data estudiantes.csv --out resultados

# Forzar k = 4 en K-Means
python pipeline_rendimiento_academico.py --data estudiantes.csv --k 4
```

| Argumento | Descripción | Por defecto |
|---|---|---|
| `--data` | Ruta del CSV con datos reales | datos sintéticos |
| `--out` | Carpeta de salida | `salidas` |
| `--k` | Forzar el número de clusters | el de máxima silueta |
| `--benchmark-knn` | Incluye KNN solo como referencia (no se exporta) | desactivado |

### Salidas generadas

| Archivo | Contenido |
|---|---|
| `modelos/clasificador_rf.joblib` | Pipeline completo del Random Forest (imputación + modelo) |
| `modelos/clasificador_logreg.joblib` | Pipeline de la Regresión Logística (imputación + escalado + modelo) |
| `modelos/kmeans_pipeline.joblib` | Pipeline de K-Means (imputación + `StandardScaler` + K-Means) |
| `modelos/metadata.json` | Variables, hiperparámetros, métricas y k elegido |
| `reportes/comparacion_modelos.csv` | Accuracy, precisión, recall, F1 y AUC por modelo |
| `reportes/importancia_variables.csv` | Importancia por impureza y por permutación |
| `reportes/coeficientes_logreg.csv` | Dirección del efecto de cada variable |
| `reportes/seleccion_k.csv` | Inercia y silueta para k = 2 a 10 |
| `reportes/perfil_clusters.csv` | Promedios por cluster, tamaño y tasa de aprobación |
| `reportes/asignacion_clusters.csv` | Cluster asignado a cada estudiante |
| `graficos/01…05_*.png` | Comparación de modelos, importancia, selección de k, perfiles y mapa PCA |

---

## 6. Formato del CSV con datos reales

Un registro por estudiante y curso. Codificación **UTF-8**, separador **coma**, punto como separador decimal. Puede descargar una plantilla desde el dashboard.

| Columna | Obligatoria | Tipo / rango | Descripción |
|---|---|---|---|
| `id_estudiante` | Opcional | texto | Identificador (si falta, se genera uno) |
| `nota_parcial_1` | Sí | 0–20 | Primera evaluación parcial |
| `nota_parcial_2` | Sí | 0–20 | Segunda evaluación parcial |
| `nota_practicas` | Sí | 0–20 | Promedio de prácticas o laboratorios |
| `promedio_tareas` | Sí | 0–20 | Promedio de tareas |
| `asistencia_pct` | Sí | 0–100 | Porcentaje de clases asistidas |
| `entregas_pct` | Sí | 0–100 | Porcentaje de trabajos entregados a tiempo |
| `participaciones` | Sí | entero ≥ 0 | Intervenciones en clase o foros |
| `horas_estudio_sem` | Sí | número ≥ 0 | Horas de estudio por semana |
| `cursos_repetidos` | Sí | entero ≥ 0 | Cursos repetidos previamente |
| `aprobo` | Una de las dos | 0 / 1 | Variable objetivo (1 = aprueba) |
| `nota_final` | Una de las dos | 0–20 | Si no hay `aprobo`, se calcula con la nota mínima aprobatoria |

Consideraciones:

- **Valores faltantes:** se imputan automáticamente con la mediana.
- **Mínimo de datos:** al menos 10 estudiantes en cada clase (aprueba / desaprueba); en la práctica se recomiendan cientos de registros.
- **Fuga de datos (data leakage):** `nota_final` nunca se usa como variable predictora. Si la predicción debe hacerse a mitad de ciclo, elimine las variables que aún no existan en ese momento (por ejemplo `nota_parcial_2`). La lista `FEATURES` está al inicio de ambos archivos `.py`.
- **Escala de notas:** el proyecto asume 0–20 y nota mínima 11. Para otra escala, ajuste `NOTA_MINIMA` y los límites de los deslizadores.

---

## 7. Resultados esperados con los datos sintéticos

Valores aproximados al ejecutar con los datos de demostración (800 estudiantes, semilla fija):

| Indicador | Valor aproximado |
|---|---|
| Tasa de aprobación | ≈ 76 % |
| AUC Random Forest / Regresión Logística | ≈ 0.99 / ≈ 0.99 |
| Accuracy Random Forest | ≈ 0.96 |
| k sugerido por silueta / por codo | 2 / 5 |

Estos valores son **muy altos porque los datos sintéticos están diseñados para ser separables**. Con datos reales se esperan métricas más modestas. Los números pueden variar ligeramente según la versión de las librerías.

---

## 8. Cómo interpretar los resultados

**Módulo supervisado**

- La métrica más importante para una alerta temprana es el **recall de "desaprueba"**: de los estudiantes que realmente van a desaprobar, qué proporción detecta el modelo.
- La **probabilidad de aprobar** se usa como semáforo (los umbrales se ajustan en la barra lateral).
- La **importancia por permutación** indica cuánto empeora la predicción al desordenar una variable. Es más confiable que la importancia por impureza.
- Los **coeficientes** de la regresión logística muestran la dirección del efecto (si la variable sube o baja la probabilidad de aprobar).
- Importancia alta **no implica causalidad**.

**Módulo no supervisado**

- El **codo** y la **silueta** orientan la elección de k y pueden discrepar. Silueta cercana a 1 indica grupos bien separados.
- Para diseñar intervenciones suele ser más útil un k entre 3 y 5 que un k = 2.
- Los nombres de los grupos ("Alto rendimiento", "En riesgo", etc.) se asignan automáticamente según el perfil; deben revisarse con criterio pedagógico.

**Por qué se descarta KNN**

- Guarda todo el dataset y calcula distancias en cada predicción (mal escalamiento).
- Es sensible a la escala y a variables irrelevantes.
- No ofrece importancia de variables ni coeficientes, que es lo que la institución necesita.
- Sus probabilidades son poco finas (múltiplos de 1/k).
- En la prueba empírica puede rendir parecido; se descarta por **explicabilidad y escalabilidad**, no por precisión.

---

## 9. Uso de los modelos exportados en otro programa

```python
import joblib
import pandas as pd

rf = joblib.load("modelos_app/clasificador_rf.joblib")        # o salidas/modelos/...
km = joblib.load("modelos_app/kmeans_pipeline.joblib")

# Las columnas deben tener exactamente estos nombres y este orden
estudiante = pd.DataFrame([{
    "nota_parcial_1": 9.0, "nota_parcial_2": 8.5, "nota_practicas": 10.0,
    "promedio_tareas": 9.5, "asistencia_pct": 62, "entregas_pct": 45,
    "participaciones": 1, "horas_estudio_sem": 3, "cursos_repetidos": 1,
}])

print("Probabilidad de aprobar:", rf.predict_proba(estudiante)[0, 1])
print("Cluster asignado:", km.predict(estudiante)[0])
```

> Cargue los modelos solo con la **misma versión de scikit-learn** con la que se guardaron.

---

## 10. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `python` no se reconoce como comando | Python no está en el PATH | Use `py` o la ruta completa (`& 'C:\Program Files\Python314\python.exe' ...`), o reinstale Python marcando *Add python.exe to PATH* |
| `-m : El término '-m' no se reconoce` | Se escribió solo `-m pip ...` sin `python` delante | Escriba el comando completo: `python -m pip install ...` |
| `ModuleNotFoundError: No module named 'seaborn'` (o `streamlit`, `plotly`, ...) | Librerías instaladas en otro Python o sin instalar | Active el entorno `.venv` y ejecute `python -m pip install -r requirements.txt` |
| La activación de `.venv` falla por políticas de scripts | Restricción de PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y volver a activar |
| `pip` falla al compilar o no encuentra una versión (Python 3.14) | La librería aún no publica paquetes para esa versión | Instale Python 3.12 o 3.13 y recree el entorno |
| `streamlit` no se reconoce como comando | No está en el PATH | Use siempre `python -m streamlit run app_rendimiento.py` |
| El navegador no se abre | Entorno sin navegador por defecto | Abra manualmente `http://localhost:8501` |
| `Port 8501 is already in use` | Otra instancia en ejecución | Cierre la otra terminal o use `--server.port 8502` |
| Pantalla "Entrenando modelos…" por mucho tiempo | Primera ejecución o dataset muy grande | Es normal en la primera carga; con más de 3000 estudiantes puede tardar más |
| `TypeError` en las anotaciones de tipo al iniciar | Python menor a 3.10 | Actualice a Python 3.10 o superior |
| Error con `st.container(key=...)` o `page_icon` | Streamlit antiguo | `python -m pip install -U streamlit` |
| El CSV no carga: "Al CSV le faltan estas columnas" | Nombres distintos a los requeridos | Compare con la plantilla descargable; respete mayúsculas y guiones bajos |
| Caracteres raros (tildes) en el CSV | Codificación distinta a UTF-8 | Guarde el CSV como *UTF-8* desde Excel (CSV UTF-8) |
| Las animaciones no se ven | Navegador antiguo o "reducir movimiento" activado en el sistema | Actualice el navegador; revise la opción de accesibilidad del sistema operativo |
| El PNG o el CSV no aparecen | Se ejecutó desde otra carpeta | Los resultados se guardan en `salidas/` dentro de la carpeta **desde la que se ejecuta**; use `--out` con una ruta explícita |

Para reiniciar el dashboard desde cero: deténgalo con `Ctrl + C`, ejecútelo de nuevo y, si hace falta, use el menú de Streamlit (esquina superior derecha) → *Clear cache*.

---

## 11. Limitaciones y consideraciones éticas

- Con datos sintéticos, los resultados son **ilustrativos**, no reflejan una institución real.
- Antes de usarlo con datos reales, valide el modelo con una **partición temporal** (entrenar con ciclos anteriores y probar con el más reciente).
- Revise sesgos y **evite variables sensibles** (género, origen, nivel socioeconómico, etc.) salvo que haya una justificación y un análisis de equidad.
- El sistema es una **herramienta de apoyo a la tutoría**, no debe usarse para sanciones o decisiones automáticas sobre los estudiantes.
- Los datos de estudiantes son personales: use datos anonimizados y cumpla la normativa de protección de datos de su institución.
- Las probabilidades del panel de riesgo provienen de validación cruzada (cada estudiante es evaluado por un modelo que no lo vio en el entrenamiento).

---

## 12. Tecnologías

Python 3 · pandas · NumPy · scikit-learn · Matplotlib / Seaborn (pipeline por terminal) · Plotly y Streamlit (dashboard) · joblib (exportación de modelos).
