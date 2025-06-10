
# 📘 MoodleLMS_Prediction: Predicción Temprana de Abandono Estudiantil

Este repositorio contiene el desarrollo de un sistema predictivo basado en Machine Learning y Deep Learning, utilizando datos de logs de Moodle LMS para identificar tempranamente a estudiantes en riesgo de reprobar la asignatura **Fundamentos de Programación**.

---

## 🎯 1. Propósito del Proyecto

El objetivo principal de este proyecto es desarrollar una solución predictiva robusta para identificar de forma proactiva a estudiantes en riesgo de reprobar la asignatura de Fundamentos de Programación. Al detectar tempranamente a estos estudiantes, se busca habilitar intervenciones pedagógicas oportunas y personalizadas, con el fin de mejorar su rendimiento académico y reducir las tasas de abandono.

### 📅 Periodos Académicos Analizados:
- **Año 2023, Semestre 1**
- **Año 2023, Semestre 2**
- **Año 2024, Semestre 1**

El modelo final busca un equilibrio óptimo entre una alta capacidad de detección (Recall para la clase 'Reprobar') y una alta eficiencia operativa (minimizando los Falsos Positivos), lo cual es crucial para la aplicabilidad en entornos educativos reales.
--

## 📂 2. Estructura del Repositorio

```
├── notebooks/
│   ├── 01_Preprocesamiento_Logs_Notas.ipynb
│   ├── 02_Preparacion_Dataset_Tabular.ipynb
│   ├── 03_Preparacion_Dataset_Secuencial.ipynb
│   └── 04_Optimizacion_Evaluacion_Modelos.ipynb
├── models/
│   └── modelo_lstm_f1_optuna_final.keras
├── plots/
│   └── lstm_f1_learning_curves.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 3. Adquisición y Preprocesamiento de Datos

### 3.1. Proceso de Preprocesamiento

El proceso general para cada período académico (2023-1, 2023-2, 2024-1) es el mismo, adaptándose a las particularidades de cada semestre:

- **Carga de Datos Crudos:**
  - Importación de logs de actividad de Moodle.
  - Importación de registros de notas.

- **Limpieza Inicial:**
  - Eliminación de duplicados y manejo de valores nulos.
  - Estandarización de nombres de usuario (eliminación de espacios, uso de minúsculas).
  - Conversión de tipos de datos a formatos adecuados (fechas, texto, categóricos).

- **Filtrado de Usuarios No Estudiantiles:**
  - Remoción de logs generados por profesores o cuentas administrativas (mediante listas de nombres y patrones).
  - Eliminación de eventos irrelevantes para el comportamiento académico.

- **Cálculo de `semana_semestre`:**
  - Se determina la semana relativa desde el inicio del semestre para cada registro de log, según el calendario académico real.

- **Agregación Semanal de Logs:**
  - Consolidación de logs diarios a nivel semanal por estudiante y semestre.

- **Cálculo de Características Acumulativas y Derivadas:**
  - Generación de variables temporales que resumen la evolución de actividad y rendimiento del estudiante.

- **Manejo Adaptativo entre Semestres/Períodos:**
  - Consideración de feriados, paros académicos u otros factores que alteran la distribución del semestre.
  - Ajuste dinámico de los cortes temporales y del cálculo de semanas.

- **Generación del `df_consolidado`:**
  - DataFrame con una fila por estudiante–semestre–semana, incluyendo todas las variables calculadas.

- **Escalado de Características:**
  - Aplicación de `StandardScaler` para normalizar variables numéricas.

- **Creación de Datasets para Modelos:**
  - Formato 2D (`muestras x características`) para modelos tradicionales (MLP, RF, etc.).
  - Formato 3D (`muestras x semanas x características`) para modelos secuenciales (LSTM, LSTM-CNN), con padding incluido.


### 3.2 Variables Derivadas Creadas
Durante el preprocesamiento, se crearon las siguientes variables clave que alimentan los modelos predictivos, reflejando tanto la actividad del estudiante en Moodle como su progreso académico:

- `semana_semestre`: Número de semana relativa desde el inicio del semestre.
- `max_days_with_access`: Máximo número de días consecutivos con actividad registrados hasta esa semana.
- `max_days_without_access`: Máximo número de días consecutivos sin actividad registrados hasta esa semana.
- `first_last_log_diff`: Diferencia en días entre el primer log del semestre y el último log registrado en la semana actual.
- `logs`: Total acumulado de registros de actividad (logs) del estudiante en el curso hasta la semana actual.
- `week_logs`: Cantidad de registros de actividad del estudiante en la semana específica.
- `daily_avg`: Promedio acumulado de logs diarios del estudiante hasta la semana actual.
- `weekly_avg`: Promedio acumulado de logs semanales del estudiante hasta la semana actual.
- `days_with_logs`: Total acumulado de días con actividad del estudiante en el curso hasta la semana actual.
- `days_with_logs_avg`: Promedio acumulado de días con actividad del estudiante por semana, hasta la semana actual.
- `days_with_logs_week`: Número de días con actividad registrados en la semana específica.
- `activity_total`, `content_total`, `other_total`, `report_total`, `system_total`: Total acumulado de logs por categorías de eventos (activity, content, other, report, system) hasta la semana actual.
- `activity_week`, `content_week`, `other_week`, `report_week`, `system_week`: Cantidad de logs por categorías de eventos en la semana específica.
- `total_weeks`: Duración total del semestre o curso en semanas (máximo observado para el grupo).
- `course_progress`: Progreso relativo del estudiante en el curso hasta la semana actual.
- `longest_streak`: La racha más larga de días consecutivos con actividad registrados hasta la semana actual.
- `promedio_ponderado`: Promedio ponderado de las calificaciones del estudiante hasta la semana actual.
- `proyeccion_nota_final`: Proyección de la nota final del estudiante en el curso, calculada hasta la semana actual.
- `aprobando`: Indicador binario (0 o 1) que señala si el estudiante está en una condición de "aprobando" el curso hasta la semana actual.
- `aprobo_semestre_real`: Variable objetivo final del modelo, indicando si el estudiante aprobó (1) o reprobó (0) el semestre.

---
---

## 🤖 4. Modelos Evaluados y Rendimiento

Se exploraron y optimizaron nueve algoritmos de Machine Learning y Deep Learning, incluyendo modelos tradicionales y redes neuronales profundas, para encontrar el modelo más efectivo. La optimización de hiperparámetros se realizó con Optuna, buscando maximizar el F1-Score de la Clase 0 ('Reprobar') para equilibrar la detección (Recall) y la precisión de las alertas (Precision).

| Modelo       | AUC    | F1-Clase0 | Recall-0 | Precision-0 | Recall-1 | Umbral |
|--------------|--------|-----------|----------|-------------|----------|--------|
| **LSTM**     | 0.9021 | 0.8050    | 0.9014   | 0.7273       | 0.7143   | 0.66   |
| LSTM-CNN     | 0.8888 | 0.7875    | 0.8873   | 0.7079       | 0.6905   | 0.60   |
| CNN          | 0.6841 | 0.6404    | 0.8028   | 0.5327       | 0.4048   | 0.66   |
| RandomForest | 0.6662 | 0.6596    | 0.8732   | 0.5299       | 0.3452   | 0.70   |
| MLP          | 0.6549 | 0.6497    | 0.9014   | 0.5079       | 0.2619   | 0.74   |
| DNN          | 0.6549 | 0.6630    | 0.8451   | 0.5455       | 0.4048   | 0.66   |
| CatBoost     | 0.6484 | 0.5926    | 0.6761   | 0.5275       | 0.4881   | 0.80   |
| SVM          | 0.6164 | 0.5978    | 0.7746   | 0.4867       | 0.3095   | 0.62   |
| KNN          | 0.5765 | 0.5870    | 0.7606   | 0.4779       | 0.2976   | 0.60   |

---

## 🏆 5. Modelo Ganador: LSTM

- **Nombre del modelo:** `modelo_lstm_f1_optuna_final.keras`
- **Umbral recomendado:** 0.66

### Métricas clave:
- **AUC:** 0.9021
- **F1 Clase 0:** 0.8050
- **Recall Clase 0:** 90.14%
- **Precision Clase 0:** 72.73%
- **Falsos positivos:** 7

---

## 📦 6. Uso del Modelo Entrenado

El modelo LSTM entrenado y optimizado se guarda en la carpeta models/.

### 📦 Cómo cargar el modelo entrenado

```python
from tensorflow.keras.models import load_model
import os

# Ruta donde se encuentra el repositorio clonado (ajustar si es necesario)
# Por ejemplo, si ejecutas el script desde la raíz del repositorio
repo_root = os.getcwd() 

# Ruta completa al archivo del modelo
model_path = os.path.join(repo_root, 'models', 'modelo_lstm_f1_optuna_final.keras')

# Cargar el modelo
modelo_cargado = load_model(model_path)

print(f"Modelo cargado exitosamente desde: {model_path}")

# Asumiendo que 'X_nueva_secuencia_escalada' es un nuevo dato preprocesado
# en el formato 3D (num_muestras, max_semanas, num_features).
# Es crucial que los nuevos datos se preprocesen y escalen EXACTAMENTE de la misma forma
# que los datos de entrenamiento (ver notebooks de preprocesamiento).

# Ejemplo de predicción para una nueva secuencia
prediction_proba = modelo_cargado.predict(X_nueva_secuencia_escalada)
prediction_class = (prediction_proba >= 0.66).astype(int)  # Usando el umbral sugerido

print(f"Probabilidad de reprobar: {prediction_proba[0][0]:.4f}")
print(f"Clasificación (0: Reprueba, 1: Aprueba): {prediction_class[0][0]}")

---

## 📚 7. Documentación y Preparación para Implantación

### Requerimientos y Preparación del Ambiente

**Requerimientos de Hardware y Software:** Se detallan los requisitos mínimos y recomendados de hardware (CPU, RAM, GPU) y software (Python 3.8+, librerías específicas: tensorflow, scikit-learn, pandas, optuna, imblearn, matplotlib) en el archivo `requirements.txt`.

**Preparación del Ambiente:** El proyecto se ha configurado para su ejecución en entornos locales mediante Python y pip. Se recomienda el uso de ambientes virtuales para gestionar las dependencias de forma aislada.

---

### Evidencia de Preparación del Ambiente

- **Repositorio Git:** Este mismo repositorio en GitHub contiene todo el código fuente, la estructura del proyecto y el historial de commits como evidencia del control de versiones.
- **Dependencias:** El archivo `requirements.txt` lista todas las librerías Python y sus versiones exactas utilizadas, asegurando la reproducibilidad del ambiente de desarrollo.
- **Artefactos del Modelo:** El modelo final `modelo_lstm_f1_optuna_final.keras` se encuentra en la carpeta `models/`. Los gráficos de curvas de aprendizaje se encuentran en `plots/`, sirviendo como evidencia del proceso de entrenamiento y diagnóstico.

---

### Documentación Adicional

**Documentación Técnica:** El código fuente está modularizado en notebooks comentados, facilitando la comprensión y mantenimiento.

**Manual de Usuario (futuro):** Se propone el desarrollo de un manual para el personal educativo que detalle cómo usar la herramienta predictiva, interpretar alertas y actuar en consecuencia.

---

## 🤝 8. Contribución

Para contribuir, haz fork del proyecto, crea una nueva rama, realiza tus cambios, y abre un Pull Request.

---

## 📄 9. Licencia

Este proyecto está bajo la licencia MIT.
