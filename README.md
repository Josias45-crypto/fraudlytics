# 🔍 Fraudlytics
Sistema inteligente de detección de fraude en transacciones financieras para uso empresarial.

Combina análisis numérico, procesamiento de lenguaje natural y modelos de Machine Learning 
avanzados para identificar transacciones sospechosas con alta precisión.

## 🚀 ¿Qué hace este sistema?
- Limpia y prepara datos históricos de transacciones financieras
- Visualiza patrones de comportamiento normal vs fraudulento
- Procesa comentarios de texto y los convierte en datos para la IA
- Entrena un modelo XGBoost optimizado con 14 variables bancarias clave
- Valida el modelo con métricas especializadas para datasets desbalanceados
- Interfaz web con login, registro, diseño dark y análisis en tiempo real
- Plantilla Excel descargable para que el usuario suba sus propios datos

## 📊 Rendimiento del modelo en producción
| Métrica | Valor |
|---------|-------|
| ROC-AUC | 99% |
| Recall | 99% |
| Precision | 99% |
| Umbral óptimo | 0.90 |
| Dataset de entrenamiento | 200,000 transacciones |
| Fraudes en entrenamiento | 0.17% (igual que datos reales) |

## 📋 Formato de datos requerido
El modelo acepta archivos CSV o Excel con estas 14 columnas:

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| monto | Monto de la transacción | 150.50 |
| hora | Hora del día (0-23) | 14 |
| tipo_transaccion | compra, retiro, transferencia, pago_servicio | compra |
| canal | web, app_movil, cajero, sucursal | app_movil |
| pais_origen | País donde se hizo la transacción | CO |
| pais_destino | País destino del dinero | CO |
| tarjeta_tipo | visa, mastercard, amex, diners | visa |
| cliente_edad | Edad del cliente | 35 |
| cliente_antiguedad_dias | Días como cliente del banco | 730 |
| transacciones_ultimas_24h | Transacciones realizadas hoy | 2 |
| monto_promedio_historico | Gasto promedio histórico | 120.00 |
| distancia_ultima_transaccion_km | Distancia desde última compra | 2.5 |
| es_horario_inusual | 1 si es de madrugada, 0 si no | 0 |
| intentos_fallidos_previos | Intentos fallidos antes | 0 |

## 🧱 Pipeline del sistema
| Paso | Archivo | Tecnologías |
|------|---------|-------------|
| 1 - Preparación de datos | `src/paso1_preparacion.py` | Pandas, NumPy, SciPy |
| 2 - Visualización | `src/paso2_visualizacion.py` | Matplotlib, Seaborn |
| 3 - Procesamiento de texto | `src/paso3_texto.py` | NLTK, Scikit-learn |
| 4 - Modelo de IA | `src/paso4_modelo.py` | TensorFlow/Keras, PyTorch |
| 5 - Validación | `src/paso5_validacion.py` | Scikit-learn |
| Modelo de producción | `src/evaluar_modelo.py` | XGBoost, Scikit-learn |
| Plantilla Excel | `src/generar_plantilla.py` | Pandas, OpenPyXL |
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

### 4. Descarga el dataset base
- Ve a kaggle.com/datasets/mlg-ulb/creditcardfraud
- Descarga `creditcard.csv` y ponlo en `data/`

## ▶️ Orden de ejecución

### Pipeline de aprendizaje
```bash
python src/paso1_preparacion.py
python src/paso2_visualizacion.py
python src/paso3_texto.py
python src/paso4_modelo.py
python src/paso5_validacion.py
```
⚠️ Cierra las ventanas de gráficas cuando aparezcan para continuar.

### Entrenar modelo de producción
```bash
python src/evaluar_modelo.py
python src/generar_plantilla.py
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
│   ├── main.py                      → Interfaz web Streamlit (dark theme)
│   └── users.json                   → Usuarios registrados
├── data/
│   ├── creditcard.csv               → Dataset base Kaggle (no en git)
│   ├── dataset_bancario.csv         → Dataset generado 200K transacciones
│   ├── prueba_transacciones.csv     → Archivo de prueba 5 filas
│   ├── prueba_transacciones.xlsx    → Archivo de prueba Excel
│   ├── plantilla_fraudlytics.xlsx   → Plantilla descargable para usuarios
│   ├── modelo_produccion.pkl        → Modelo XGBoost entrenado
│   ├── scaler_produccion.pkl        → Scaler de normalización
│   ├── features_produccion.pkl      → Features del modelo
│   ├── umbral_produccion.pkl        → Umbral óptimo
│   └── encoders.pkl                 → Encoders de variables categóricas
├── src/
│   ├── paso1_preparacion.py
│   ├── paso2_visualizacion.py
│   ├── paso3_texto.py
│   ├── paso4_modelo.py
│   ├── paso5_validacion.py
│   ├── evaluar_modelo.py            → Entrena modelo de producción
│   ├── generar_plantilla.py         → Genera plantilla Excel
│   ├── generar_dataset_bancario.py  → Genera dataset bancario
│   ├── entrenar_modelo.py
│   ├── entrenar_ieee.py
│   ├── optimizar_modelo.py
│   ├── modelo_produccion_final.py
│   └── ajustar_umbral.py
├── requirements.txt
└── README.md
```

## 📊 Dataset
| Dataset | Fuente | Transacciones | Fraudes |
|---------|--------|--------------|---------|
| Credit Card Fraud | Kaggle (mlg-ulb) | 284,807 | 0.17% |
| Dataset bancario generado | Distribuciones reales creditcard | 200,000 | 0.17% |

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
- openpyxl