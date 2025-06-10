# MoodleLMS_code


Este repositorio contiene el desarrollo, experimentación y evaluación de modelos de Machine Learning y Deep Learning aplicados a datos de logs de Moodle LMS para predecir el rendimiento académico de los estudiantes.

## Propósito del Proyecto

El objetivo principal de este proyecto es desarrollar una solución predictiva robusta para identificar de forma proactiva a estudiantes en riesgo de reprobar la asignatura de **Fundamentos de Programación**. Al detectar tempranamente a estos estudiantes, se busca habilitar intervenciones pedagógicas oportunas y personalizadas, con el fin de mejorar su rendimiento académico y reducir las tasas de abandono.

Se trabajó con datos de los siguientes períodos académicos:
* **Fundamentos de Programación - Año 2023, Semestre 1**
* **Fundamentos de Programación - Año 2023, Semestre 2**
* **Fundamentos de Programación - Año 2024, Semestre 1**

El modelo final busca un equilibrio óptimo entre una alta capacidad de detección (`Recall` para la clase 'Reprobar') y una alta eficiencia operativa (minimizando los `Falsos Positivos`), lo cual es crucial para la aplicabilidad en entornos educativos reales.

## 📂 Estructura del Repositorio

├── notebooks/
│   ├── Preprocesamiento.ipynb
│   ├── CreacionVariables.ipynb
│   └── Creacion_Modelos.ipynb
├── models/
│   └── modelo_lstm_f1_optuna_final.keras
├── README.md
```

---

## 🔧 Requerimientos del Sistema

### Mínimos:
- Python 3.9 o superior
- 4 GB de RAM
- 1 CPU
- 2 GB de espacio libre
- pip y Git

###  Recomendados:
- Python 3.10
- 8 GB de RAM
- CPU con 4 núcleos
- Entorno virtual (`venv` o `conda`)
- Jupyter Notebook o VS Code

---

## ⚙️ Preparación del Ambiente

. Clona el repositorio:
```bash
git clone https://github.com/MaxBKoscina/MoodleLMS_code.git
cd MoodleLMS_code
```

2. Crea un entorno virtual:
```bash
python -m venv venv
.env\Scriptsctivate    # Windows
source venv/bin/activate  # Linux/Mac
```

3. Instala dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecuta Jupyter:
```bash
jupyter notebook
```

# 📈 Adquisición y Preprocesamiento de Datos

La fase de preprocesamiento es crítica y se detalla en el notebook `01_Preprocesamiento_Logs_Notas.ipynb`. A grandes rasgos, el proceso incluye:

1.  **Carga de Datos Crudos:** Se cargan los archivos de logs de Moodle y los registros de notas de cada semestre (ej. `Logs-FundaPrograma-202X-X.xlsx`, `[2023-1]_Notas.xlsx`).
2.  **Limpieza Inicial:** Eliminación de duplicados, manejo de valores nulos, y estandarización de columnas clave (ej. nombres de usuario a minúsculas, conversión a tipos de datos apropiados).
3.  **Filtrado de Eventos de Profesor y Usuarios Administrativos:** Se eliminan los registros generados por la actividad de profesores o cuentas de administración para asegurar que el comportamiento refleje únicamente la interacción de los estudiantes.
4.  **Cálculo de `semana_semestre`:** Se calcula la semana relativa dentro de cada semestre para cada entrada de log, basándose en la fecha de inicio del período académico correspondiente.
5.  **Agregación Semanal de Logs:** Los logs diarios se agregan a un nivel semanal, calculando características como el total de logs por semana, días con actividad, y logs por categoría de evento (`activity_week`, `content_week`, etc.).
6.  **Cálculo de Características Acumulativas y Derivadas:** Se generan métricas que evolucionan a lo largo del semestre, como `logs` totales acumulados, promedios de actividad (`daily_avg`, `weekly_avg`), rachas (`longest_streak`), y métricas de progreso (`course_progress`). Crucialmente, se integran variables de rendimiento académico como `promedio_ponderado` y `proyeccion_nota_final`, las cuales reflejan el estado del estudiante `hasta el momento` en cada semana, permitiendo a los modelos secuenciales aprender de estas trayectorias.
7.  **Manejo Adaptativo entre Semestres/Períodos:**
    * El proceso de preprocesamiento fue diseñado para ser **generalizable y aplicable a diferentes semestres (2023-1, 2023-2, 2024-1)**.
    * Se tuvo especial consideración en cómo la **duración de las semanas para hitos importantes** (como certificaciones o evaluaciones) puede variar entre semestres debido a factores externos (ej. feriados, paros académicos, ajustes de calendario). Los cálculos de `semana_semestre` y el corte de datos se adaptan a la duración real de cada período.
    * Este `df_consolidado` final, que contiene una observación por estudiante-semestre-semana con todas las características calculadas y actualizadas temporalmente, es la base para la creación de los datasets para los modelos. El procedimiento es consistente para todos los semestres, asegurando la comparabilidad de las características.
8.  **Escalado de Características:** Las características numéricas finales en `df_consolidado` se escalan (ej. usando `StandardScaler`) para optimizar el rendimiento de los modelos de Machine Learning.

## 🧠 Modelos Evaluados y Rendimiento


## 📊 Resultados de Modelos

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
## 📦 Modelo Entrenado

Modelo LSTM optimizado con Optuna para F1-score:

```python
from tensorflow.keras.models import load_model
modelo = load_model('models/modelo_lstm_f1_optuna_final.keras')
```

---


## 📘 Manual de Usuario

1. Ejecuta `Preprocesamiento.ipynb`
2. Luego `CreacionVariables.ipynb`
3. Finalmente `Creacion_Modelos.ipynb`
4. O carga el modelo entrenado desde `models/`

---


**Maximiliano Zvonko Baranda Koscina**  
Ingeniería Civil Informática – Universidad de Valparaíso  
Seminario de Título I (2025)

