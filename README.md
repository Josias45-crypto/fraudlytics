# 🔍 Fraudlytics
Sistema inteligente de análisis y detección de fraude en transacciones financieras.

Combina análisis numérico y procesamiento de lenguaje natural para identificar 
transacciones sospechosas, aprendiendo patrones tanto en montos, fechas y categorías 
como en los comentarios de los usuarios.

## 🚀 ¿Qué hace este sistema?
- Limpia y prepara datos históricos de transacciones financieras
- Visualiza patrones de comportamiento normal vs fraudulento
- Procesa comentarios de texto y los convierte en datos para la IA
- Entrena un modelo multimodal que combina datos numéricos y texto
- Valida el modelo con métricas especializadas para datasets desbalanceados
- Interfaz web para analizar transacciones sin tocar la terminal

## 🧱 Pipeline del sistema
| Paso | Archivo | Tecnologías |
|------|---------|-------------|
| 1 - Preparación de datos | `src/paso1_preparacion.py` | Pandas, NumPy, SciPy |
| 2 - Visualización | `src/paso2_visualizacion.py` | Matplotlib, Seaborn |
| 3 - Procesamiento de texto | `src/paso3_texto.py` | NLTK, Scikit-learn |
| 4 - Modelo de IA | `src/paso4_modelo.py` | TensorFlow/Keras, PyTorch |
| 5 - Validación | `src/paso5_validacion.py` | Scikit-learn |
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

### 4. Descarga el dataset
- Ve a kaggle.com/datasets/mlg-ulb/creditcardfraud
- Descarga el archivo `creditcard.csv`
- Colócalo dentro de la carpeta `data/`

## ▶️ Uso

### Correr el pipeline completo
Ejecuta los pasos en orden desde la raíz del proyecto:
```bash
python src/paso1_preparacion.py
python src/paso2_visualizacion.py
python src/paso3_texto.py
python src/paso4_modelo.py
python src/paso5_validacion.py
```
⚠️ Cierra las ventanas de gráficas cuando aparezcan para que el programa continúe.

### Correr la interfaz web
```bash
streamlit run app/main.py
```
Se abrirá automáticamente en tu navegador. Sube el archivo `creditcard.csv` 
desde el panel izquierdo y explora las 4 pestañas.

## 📁 Estructura del proyecto
```
fraudlytics/
├── app/
│   └── main.py
├── data/
│   └── creditcard.csv
├── notebooks/
├── src/
│   ├── paso1_preparacion.py
│   ├── paso2_visualizacion.py
│   ├── paso3_texto.py
│   ├── paso4_modelo.py
│   └── paso5_validacion.py
├── requirements.txt
└── README.md
```

## 📊 Dataset
- **Fuente:** Credit Card Fraud Detection - Kaggle (mlg-ulb)
- **Tamaño:** 284,807 transacciones
- **Balance:** 99.8% normales | 0.2% fraudes

## 🧰 Tecnologías
- Python 3.12
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn
- NLTK, Scikit-learn
- TensorFlow/Keras
- PyTorch
- Streamlit