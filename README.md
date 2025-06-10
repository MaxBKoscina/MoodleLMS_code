MoodleLMS_Prediction: Predicción Temprana de Abandono Estudiantil
Este repositorio contiene el desarrollo de un sistema predictivo basado en Machine Learning y Deep Learning, utilizando datos de logs de Moodle LMS para identificar tempranamente a estudiantes en riesgo de reprobar.

🎯 1. Propósito del Proyecto
El objetivo principal de este proyecto es desarrollar una solución predictiva robusta para identificar de forma proactiva a estudiantes en riesgo de reprobar la asignatura de Fundamentos de Programación. Al detectar tempranamente a estos estudiantes, se busca habilitar intervenciones pedagógicas oportunas y personalizadas, con el fin de mejorar su rendimiento académico y reducir las tasas de abandono.

Se trabajó con datos de los siguientes períodos académicos, lo que permitió una evaluación robusta de la generalización de los modelos:

Año 2023, Semestre 1

Año 2023, Semestre 2

Año 2024, Semestre 1

El modelo final busca un equilibrio óptimo entre una alta capacidad de detección (Recall para la clase 'Reprobar') y una alta eficiencia operativa (minimizando los Falsos Positivos), lo cual es crucial para la aplicabilidad en entornos educativos reales.

📂 2. Estructura del Repositorio
La organización del proyecto en GitHub es la siguiente:

├── notebooks/
│   ├── 01_Preprocesamiento_Logs_Notas.ipynb   # Carga de datos crudos, limpieza inicial, y preparación del df_consolidado.
│   ├── 02_Preparacion_Dataset_Tabular.ipynb   # Preparación del dataset para modelos tabulares (MLP, DNN, CNN).
│   ├── 03_Preparacion_Dataset_Secuencial.ipynb # Creación de secuencias temporales para modelos recurrentes (LSTM, LSTM-CNN).
│   └── 04_Optimizacion_Evaluacion_Modelos.ipynb # Contiene el código para la optimización (Optuna) y evaluación de todos los modelos.
├── models/
│   └── modelo_lstm_f1_optuna_final.keras # Modelo LSTM ganador entrenado y guardado.
│   └── # Otros modelos .keras o .pkl entrenados si se subieron (ej. modelo_rf_optuna.pkl).
├── data/
│   └── raw/                              # Carpeta para almacenar archivos de datos crudos (logs y notas).
│       ├── Logs-FundaPrograma-2023-1.xlsx # Ejemplo de un archivo de logs
│       ├── Logs-FundaPrograma-2023-2.xlsx
│       ├── Logs-FundaPrograma-2024-1.xlsx
│       └── [2023-1]_Notas.xlsx           # Ejemplo de un archivo de notas
├── plots/                                # Contiene gráficos de curvas de aprendizaje y otros resultados visuales.
│   └── lstm_f1_learning_curves.png       # Ejemplo: Gráfica de entrenamiento del modelo ganador.
├── .gitignore                            # Archivo para especificar qué ignorar en Git (ej. entornos virtuales, archivos de datos muy grandes).
├── requirements.txt                      # Listado de dependencias de Python del proyecto.
└── README.md                             # Este archivo.

📈 3. Adquisición y Preprocesamiento de Datos
La fase de preprocesamiento es fundamental y se detalla en los notebooks bajo notebooks/. Se implementó un pipeline para transformar los datos de logs de Moodle y los registros de notas en un formato apto para el modelado predictivo.

3.1. Proceso de Preprocesamiento
El proceso general para cada período académico (2023-1, 2023-2, 2024-1) es el mismo, adaptándose a las particularidades de cada semestre:

Carga de Datos Crudos: Importación de logs de actividad de Moodle y registros de notas.

Limpieza Inicial:

Eliminación de duplicados y manejo de valores nulos.

Estandarización de nombres de usuario (eliminar espacios, minúsculas).

Conversión de tipos de datos a formatos adecuados (fechas, texto, categóricos).

Filtrado de Usuarios No Estudiantiles:

Remoción de logs generados por profesores o cuentas administrativas (mediante listas de nombres y patrones).

Eliminación de eventos de logs irrelevantes para el comportamiento del estudiante.

Cálculo de semana_semestre: Se calcula la semana relativa dentro de cada semestre para cada registro de log, basándose en la fecha de inicio del período académico correspondiente.

Agregación Semanal de Logs: Los logs diarios se consolidan a un nivel semanal por estudiante y semestre, resumiendo la actividad del estudiante en cada semana.

Cálculo de Características Acumulativas y Derivadas: Se generan un conjunto rico de variables que evolucionan a lo largo del semestre, capturando la trayectoria del estudiante tanto en actividad como en rendimiento.

Manejo Adaptativo entre Semestres/Períodos:

El preprocesamiento fue diseñado para ser generalizable a través de diferentes semestres.

Se tuvo especial consideración en cómo la duración de las semanas para hitos importantes (como certificaciones o evaluaciones) puede variar entre semestres debido a factores externos (ej. feriados, paros académicos, ajustes de calendario). Los cálculos de semana_semestre y los cortes de datos se ajustan dinámicamente a la duración real de cada período.

Este df_consolidado final, que contiene una observación por estudiante-semestre-semana con todas las características calculadas y actualizadas temporalmente, es la base para la creación de los datasets para los modelos. El procedimiento es consistente para todos los semestres, asegurando la comparabilidad de las características.

Escalado de Características: Las características numéricas finales en el df_consolidado se escalan (ej. usando StandardScaler) para optimizar el rendimiento de los modelos de Machine Learning.

Creación de Datasets para Modelos: El df_consolidado se transforma en el formato 2D (muestras x características) para modelos tabulares y en formato 3D (muestras x pasos_de_tiempo x características) para modelos secuenciales (LSTMs), incluyendo padding para estandarizar la longitud de las secuencias.

3.2. Variables Derivadas Creadas
Durante el preprocesamiento, se crearon las siguientes variables clave que alimentan los modelos predictivos, reflejando tanto la actividad del estudiante en Moodle como su progreso académico:

semana_semestre: Número de semana relativa desde el inicio del semestre.

max_days_with_access: Máximo número de días consecutivos con actividad registrados hasta esa semana.

max_days_without_access: Máximo número de días consecutivos sin actividad registrados hasta esa semana.

first_last_log_diff: Diferencia en días entre el primer log del semestre y el último log registrado en la semana actual.

logs: Total acumulado de registros de actividad (logs) del estudiante en el curso hasta la semana actual.

week_logs: Cantidad de registros de actividad del estudiante en la semana específica.

daily_avg: Promedio acumulado de logs diarios del estudiante hasta la semana actual.

weekly_avg: Promedio acumulado de logs semanales del estudiante hasta la semana actual.

days_with_logs: Total acumulado de días con actividad del estudiante en el curso hasta la semana actual.

days_with_logs_avg: Promedio acumulado de días con actividad del estudiante por semana, hasta la semana actual.

days_with_logs_week: Número de días con actividad registrados en la semana específica.

activity_total, content_total, other_total, report_total, system_total: Total acumulado de logs por categorías de eventos (activity, content, other, report, system) hasta la semana actual.

activity_week, content_week, other_week, report_week, system_week: Cantidad de logs por categorías de eventos en la semana específica.

total_weeks: Duración total del semestre o curso en semanas (máximo observado para el grupo).

course_progress: Progreso relativo del estudiante en el curso hasta la semana actual.

longest_streak: La racha más larga de días consecutivos con actividad registrados hasta la semana actual.

promedio_ponderado: Promedio ponderado de las calificaciones del estudiante hasta la semana actual.

proyeccion_nota_final: Proyección de la nota final del estudiante en el curso, calculada hasta la semana actual.

aprobando: Indicador binario (0 o 1) que señala si el estudiante está en una condición de "aprobando" el curso hasta la semana actual.

aprobo_semestre_real: Variable objetivo final del modelo, indicando si el estudiante aprobó (1) o reprobó (0) el semestre (esta etiqueta se asocia a la secuencia completa del estudiante en ese semestre).
🧠 4. Modelos Evaluados y Rendimiento
Se exploraron y optimizaron nueve algoritmos de Machine Learning y Deep Learning, incluyendo modelos tradicionales y redes neuronales profundas, para encontrar el modelo más efectivo. La optimización de hiperparámetros se realizó con Optuna, buscando maximizar el F1-Score de la Clase 0 ('Reprobar') para equilibrar la detección (Recall) y la precisión de las alertas (Precision).

| Modelo       | AUC    | F1-Clase0 | Recall-0 | Precision-0 | Recall-1 | Umbral |
|--------------|--------|-----------|----------|-------------|----------|--------|
| **LSTM**     | 0.9021 | 0.8050   | 0.9014  | 0.7273     | 0.7143   | 0.66   |
| LSTM-CNN     | 0.8888 | 0.7875   | 0.8873  | 0.7079     | 0.6905   | 0.60   |
| CNN          | 0.6841 | 0.6404   | 0.8028  | 0.5327     | 0.4048   | 0.66   |
| RandomForest | 0.6662 | 0.6596   | 0.8732  | 0.5299     | 0.3452   | 0.70   |
| MLP          | 0.6549 | 0.6497   | 0.9014  | 0.5079     | 0.2619   | 0.74   |
| DNN          | 0.6549 | 0.6630   | 0.8451  | 0.5455     | 0.4048   | 0.66   |
| CatBoost     | 0.6484 | 0.5926   | 0.6761  | 0.5275     | 0.4881   | 0.80   |
| SVM          | 0.6164 | 0.5978   | 0.7746  | 0.4867     | 0.3095   | 0.62   |
| KNN          | 0.5765 | 0.5870   | 0.7606  | 0.4779     | 0.2976   | 0.60   |

---
0,60

🏆 5. Modelo Ganador
El modelo seleccionado como el más óptimo para la predicción temprana de abandono estudiantil es la Red Neuronal Recurrente (LSTM). Este modelo demostró el mejor rendimiento general y un balance excepcional entre la capacidad de detección y la eficiencia operativa.

Nombre del Modelo: modelo_lstm_f1_optuna_final.keras

Umbral de Operación Sugerido: 0,66

Métricas Clave en el Umbral Sugerido:

AUC: 0,9021

F1-Score (Clase 0): 0,8050

Recall (Clase 0): 0,9014 (¡90,14% de los estudiantes en riesgo detectados!)

Precisión (Clase 0): 0,7273

Falsos Positivos (FP-0): 7 (¡Solo 7 falsas alarmas para 90,14% de detección!)

📦 6. Uso del Modelo Entrenado
El modelo LSTM entrenado y optimizado se guarda en la carpeta models/.

Cómo cargarlo:
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

Cómo usarlo para predecir:
# Asumiendo que 'X_nueva_secuencia_escalada' es un nuevo dato preprocesado
# en el formato 3D (num_muestras, max_semanas, num_features).
# Es crucial que los nuevos datos se preprocesen y escalen EXACTAMENTE de la misma forma
# que los datos de entrenamiento (ver notebooks de preprocesamiento).

# Ejemplo de predicción para una nueva secuencia
# prediction_proba = modelo_cargado.predict(X_nueva_secuencia_escalada)
# prediction_class = (prediction_proba >= 0.66).astype(int) # Usando el umbral sugerido

# print(f"Probabilidad de reprobar: {prediction_proba[0][0]:.4f}")
# print(f"Clasificación (0: Reprueba, 1: Aprueba): {prediction_class[0][0]}")

📚 7. Documentación y Preparación para Implantación
Requerimientos y Preparación del Ambiente
Requerimientos de Hardware y Software: Se detallan los requisitos mínimos y recomendados de hardware (CPU, RAM, GPU) y software (Python 3.8+, librerías específicas: tensorflow, scikit-learn, pandas, optuna, imblearn, matplotlib) en requirements.txt.

Preparación del Ambiente: El proyecto se ha configurado para su ejecución en entornos locales mediante Python y pip. Se recomienda el uso de ambientes virtuales para gestionar las dependencias de forma aislada.

Evidencia de Preparación del Ambiente
Repositorio Git: Este mismo repositorio en GitHub (https://github.com/tu_usuario/MoodleLMS_Prediction.git) sirve como evidencia central del control de versiones, conteniendo todo el código fuente, la estructura del proyecto y el historial de commits.

Dependencias: El archivo requirements.txt ubicado en la raíz del repositorio lista todas las librerías Python y sus versiones exactas utilizadas, asegurando la reproducibilidad del ambiente de desarrollo.

Artefactos del Modelo: El modelo ganador modelo_lstm_f1_optuna_final.keras se encuentra en la carpeta models/. Los gráficos de las curvas de aprendizaje (ej., lstm_f1_learning_curves.png) se encuentran en la carpeta plots/, sirviendo como evidencia del proceso de entrenamiento y diagnóstico.

Documentación Adicional (Manual de Uso)
Documentación Técnica (Código): El código fuente está modularizado en notebooks claros y funciones bien comentadas, facilitando la comprensión y el mantenimiento.

Manual de Usuario (Propuesta de Trabajo Futuro): Para una implantación completa, se propone desarrollar un manual de usuario para el personal educativo, detallando el uso de la solución predictiva (cómo ingresar datos, interpretar alertas, etc.).

🤝 8. Contribución
Para contribuir a este proyecto, por favor, sigue las prácticas estándar de Git (fork, branch, commit, pull request).

📄 9. Licencia
Este proyecto está bajo la licencia [Tu Licencia, ej. MIT].
