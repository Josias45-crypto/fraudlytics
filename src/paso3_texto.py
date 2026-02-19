# ============================================================
# FRAUDLYTICS - PASO 3: Procesamiento de Texto
# NLTK y Scikit-learn
# ============================================================

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter

# Descarga de recursos NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')

# ============================================================
# CARGA Y SIMULACIÓN DE COMENTARIOS
# ============================================================

print("📦 Cargando datos procesados...")
df = pd.read_csv("data/datos_procesados.csv")
print(f"✅ Datos cargados: {df.shape[0]} filas")

print("\n💬 Generando comentarios simulados de transacciones...")

comentarios_normales = [
    "pago en supermercado compra semanal",
    "transferencia a cuenta de ahorros personal",
    "pago de servicios publicos agua y luz",
    "compra en restaurante almuerzo de trabajo",
    "retiro en cajero automatico del banco",
    "pago de suscripcion mensual streaming",
    "compra en farmacia medicamentos",
    "pago de transporte uber viaje al aeropuerto",
    "compra en tienda ropa temporada",
    "pago de internet y telefonia hogar",
]

comentarios_fraude = [
    "transferencia urgente cuenta desconocida extranjero",
    "compra sospechosa monto elevado madrugada",
    "retiro multiple cajero diferente ciudad",
    "pago no reconocido tarjeta clonada posible",
    "transaccion extraña horario inusual monto alto",
    "compra electronica costosa sin historial previo",
    "transferencia internacional cuenta nueva sin verificar",
    "movimiento inusual patron diferente usuario",
    "pago duplicado mismo comercio minutos despues",
    "actividad nocturna sospechosa monto irregular",
]

np.random.seed(42)
comentarios = []
for _, row in df.iterrows():
    if row["Class"] == 0:
        comentarios.append(np.random.choice(comentarios_normales))
    else:
        comentarios.append(np.random.choice(comentarios_fraude))

df["comentario"] = comentarios
print("✅ Comentarios generados")

# ============================================================
# BLOQUE A - NLTK: Limpieza y análisis de texto
# ============================================================

print("\n🔤 Procesando texto con NLTK...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    tokens = word_tokenize(texto.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

df["comentario_limpio"] = df["comentario"].apply(limpiar_texto)
print("✅ Texto limpiado y lematizado")

# Análisis de N-gramas
print("\n📊 Analizando N-gramas más frecuentes...")
todos_los_tokens = " ".join(df["comentario_limpio"].values).split()
bigramas = list(ngrams(todos_los_tokens, 2))
trigramas = list(ngrams(todos_los_tokens, 3))

top_bigramas = Counter(bigramas).most_common(5)
top_trigramas = Counter(trigramas).most_common(5)

print(f"✅ Top 5 bigramas: {top_bigramas}")
print(f"✅ Top 5 trigramas: {top_trigramas}")

# ============================================================
# BLOQUE B - SKLEARN: TF-IDF y reducción LSA
# ============================================================

print("\n🔢 Aplicando TF-IDF...")
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(df["comentario_limpio"])
print(f"✅ Matriz TF-IDF: {X_tfidf.shape}")

print("\n📉 Reduciendo dimensionalidad con LSA (TruncatedSVD)...")
lsa = TruncatedSVD(n_components=50, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)
print(f"✅ Matriz LSA reducida a: {X_lsa.shape}")
print(f"✅ Varianza explicada: {lsa.explained_variance_ratio_.sum():.2%}")

# ============================================================
# GUARDADO
# ============================================================

print("\n💾 Guardando resultados...")
df_lsa = pd.DataFrame(X_lsa, columns=[f"lsa_{i}" for i in range(50)])
df_final = pd.concat([df.reset_index(drop=True), df_lsa], axis=1)
df_final.to_csv("data/datos_con_texto.csv", index=False)
print("✅ Guardado en data/datos_con_texto.csv")

print("\n🎉 Paso 3 completado exitosamente!")