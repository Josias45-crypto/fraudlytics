# 🔍 Fraudlytics
Sistema inteligente de detección de fraude en transacciones financieras para uso empresarial.

Combina análisis numérico, procesamiento de lenguaje natural y modelos de Machine Learning 
avanzados para identificar transacciones sospechosas con alta precisión.

## 🚀 ¿Qué hace este sistema?
- Limpia y prepara datos históricos de transacciones financieras
- Visualiza patrones de comportamiento normal vs fraudulento
- Procesa comentarios de texto y los convierte en datos para la IA
- Entrena un modelo multimodal que combina datos numéricos y texto
- Optimiza el modelo con SMOTE, XGBoost y ajuste de umbral para producción
- Valida el modelo con métricas especializadas para datasets desbalanceados
- Interfaz web con login, registro y análisis en tiempo real

## 📊 Rendimiento del modelo en producción
| Métrica | Valor |
|---------|-------|
| ROC-AUC | 94.9% |
| Recall | 71.4% |
| Precision | 53.2% |
| Umbral óptimo | 0.70 |
| Dataset de entrenamiento | 590,540 transacciones reales |

## 🧱 Pipeline del sistema
| Paso | Archivo | Tecnologías |
|------|---------|-------------|
| 1 - Preparación de datos | `src/paso1_preparacion.py` | Pandas, NumPy, SciPy |
| 2 - Visualización | `src/paso2_visualizacion.py` | Matplotlib, Seaborn |
| 3 - Procesamiento de texto | `src/paso3_texto.py` | NLTK, Scikit-learn |
| 4 - Modelo de IA | `src/paso4_modelo.py` | TensorFlow/Keras, PyTorch |
| 5 - Validación | `src/paso5_validacion.py` | Scikit-learn |
| Entrenamiento producción | `src/entrenar_ieee.py` | XGBoost, IEEE-CIS |
| Optimización umbral | `src/ajustar_umbral.py` | Scikit-learn |
| App web | `app/main.py` | Streamlit |

## 🛠️ Instalación

### 1. Clona el repositorio
```bash
git clone https://github.com/Josias45-crypto/fraudlytics.git
cd fraudlytics
```

### 2. Crea y activa el entorno virtual
```bash
# Windows
python -m venv venv
venv\Scripts\Activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Instala las dependencias
```bash
pip install -r requirements.txt
pip install torch
```

### 4. Descarga los datasets
**Dataset base (pipeline de aprendizaje):**
- Ve a kaggle.com/datasets/mlg-ulb/creditcardfraud
- Descarga `creditcard.csv` y ponlo en `data/`

**Dataset de producción (modelo real):**
- Ve a kaggle.com/competitions/ieee-fraud-detection/data
- Acepta las reglas de la competencia
- Descarga `train_transaction.csv` y `train_identity.csv` y ponlos en `data/`

## ▶️ Uso

### Correr el pipeline completo de aprendizaje
```bash
python src/paso1_preparacion.py
python src/paso2_visualizacion.py
python src/paso3_texto.py
python src/paso4_modelo.py
python src/paso5_validacion.py
```
⚠️ Cierra las ventanas de gráficas cuando aparezcan para que el programa continúe.

### Entrenar el modelo de producción
```bash
python src/entrenar_ieee.py
python src/ajustar_umbral.py
```

### Correr la interfaz web
```bash
streamlit run app/main.py
```
Se abrirá en tu navegador. Credenciales por defecto: **admin / admin123**

## 📁 Estructura del proyecto
```
fraudlytics/
├── app/
│   ├── main.py
│   └── users.json
├── data/
│   ├── creditcard.csv
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── modelo_produccion.pkl
├── notebooks/
├── src/
│   ├── paso1_preparacion.py
│   ├── paso2_visualizacion.py
│   ├── paso3_texto.py
│   ├── paso4_modelo.py
│   ├── paso5_validacion.py
│   ├── entrenar_modelo.py
│   ├── entrenar_ieee.py
│   ├── optimizar_modelo.py
│   ├── modelo_produccion_final.py
│   ├── evaluar_modelo.py
│   └── ajustar_umbral.py
├── requirements.txt
└── README.md
```

## 📊 Datasets
| Dataset | Fuente | Transacciones | Fraudes |
|---------|--------|--------------|---------|
| Credit Card Fraud | Kaggle (mlg-ulb) | 284,807 | 0.17% |
| IEEE-CIS Fraud Detection | Kaggle (IEEE) | 590,540 | 3.50% |

## 🧰 Tecnologías
- Python 3.12
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn
- NLTK, Scikit-learn
- TensorFlow/Keras
- PyTorch
- XGBoost
- Streamlit
- bcrypt