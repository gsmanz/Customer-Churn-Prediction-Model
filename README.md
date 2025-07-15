# 🧠 Predicción de Churn (Fuga de Clientes)

## 1. 📌 Descripción

Este proyecto tiene como objetivo desarrollar un modelo de *machine learning* para predecir la probabilidad de que un cliente abandone un servicio (*churn*).  
Se ha llevado a cabo un completo proceso de limpieza, preprocesamiento y modelado de datos.

La solución implementada incluye la construcción de un **pipeline** que automatiza las etapas de imputación, transformación, reducción de dimensionalidad y entrenamiento del modelo. Se emplea un enfoque modular para facilitar la mantenibilidad y la reproducibilidad del modelo.

---

## 2. 🗂️ Contenido del Repositorio

La estructura del repositorio es la siguiente:

📁 data/ # Contiene el dataset original
📁 notebooks/ # Incluye notebooks para exploración, limpieza, pipelines y modelado
📄 churn_model.pkl # Modelo entrenado serializado
📄 README.md # Este archivo
📄 .gitignore # Archivos a ignorar por Git


---

## 3. 🛠️ Librerías Utilizadas

El proyecto emplea las siguientes librerías y frameworks principales:

- **Análisis y visualización:** `pandas`, `numpy`, `matplotlib`, `seaborn`  
- **Preprocesamiento:** `scikit-learn`, `imblearn`  
- **Modelado:** `scikit-learn`, `XGBoost`, `CatBoost`  
- **Interpretabilidad:** `shap`  
- **Persistencia:** `pickle`  
- **Warnings:** `warnings`  

---

## 4. ⚙️ Metodología de Preprocesamiento

El pipeline de preprocesamiento y modelado se estructura en los siguientes pasos:

### 🔹 Imputación de valores nulos
- Imputación constante con valor `'Unknown'` para variables categóricas con alta proporción de nulos (`area`, `ownrent`, etc.).
- Imputación con `0.0`, mediana o valor más frecuente, según el tipo y distribución de la variable.

### 🔹 Reducción de cardinalidad
- Implementación de **CumsumVarFilter**, que agrupa categorías poco frecuentes bajo la etiqueta `'other'` según un umbral acumulado de frecuencia (90%).

### 🔹 Transformación de variables
- **Categóricas:** One-Hot Encoding (`OneHotEncoder`)  
- **Numéricas:** Normalización (`StandardScaler`)

### 🔹 Reducción de dimensionalidad
- **Alta correlación:** Eliminación de variables altamente correlacionadas entre sí (`> 0.9`), conservando la que mejor se correlaciona con el objetivo.
- **Baja varianza:** Eliminación de variables con varianza menor a un umbral definido (`0.05`)

---

## 5. 🤖 Modelado

Se probaron múltiples algoritmos de clasificación, entre ellos:

- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `XGBClassifier`
- `CatBoostClassifier` (modelo final seleccionado)

El modelo final fue entrenado dentro de un **pipeline completo** que incluye todo el flujo de preprocesamiento mencionado anteriormente.

**Métrica principal de evaluación:**  
`f1_score` sobre el conjunto de test:

> **F1-score (X_test):** `0.839`

---

## 6. 🧮 Interpretabilidad

Se empleó la librería **SHAP** para interpretar las predicciones del modelo y entender qué variables tienen mayor peso en la predicción del churn.

---

## 7. ♻️ Reproducibilidad

El pipeline completo ha sido construido con `scikit-learn` y puede ser reutilizado para nuevos datasets simplemente cargando el modelo entrenado:

```python
import pickle

with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)
