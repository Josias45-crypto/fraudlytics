# Fraudlytics 🔍
Sistema inteligente de análisis y detección de fraude en transacciones financieras.

## ¿Qué hace?
Combina análisis numérico y procesamiento de lenguaje natural para identificar 
transacciones sospechosas, aprendiendo patrones tanto en montos y fechas 
como en los comentarios de los usuarios.

## Pipeline del sistema
1. **Paso 1** - Preparación de datos (Pandas, NumPy, SciPy)
2. **Paso 2** - Visualización de patrones (Matplotlib, Seaborn)
3. **Paso 3** - Procesamiento de texto (NLTK, Scikit-learn)
4. **Paso 4** - Modelo multimodal de IA (TensorFlow/Keras + PyTorch)
5. **Paso 5** - Validación robusta (Scikit-learn)

## Tecnologías
- Python 3.12
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn
- NLTK, Scikit-learn
- TensorFlow/Keras
- PyTorch

## Instalación
```bash
git clone https://github.com/Josias45-crypto/fraudlytics.git
cd fraudlytics
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
```

## Uso
```bash
python src/paso1_preparacion.py
python src/paso2_visualizacion.py
python src/paso3_texto.py
python src/paso4_modelo.py
python src/paso5_validacion.py
```

## Dataset
Credit Card Fraud Detection - Kaggle (mlg-ulb)
284,807 transacciones | 0.2% fraudes