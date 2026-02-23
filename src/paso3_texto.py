# ============================================================
# FRAUDLYTICS - PASO 3: Procesamiento de Lenguaje Natural
# NLTK + Scikit-learn: N-gramas, TF-IDF, LSA
# Adaptado al dataset bancario con comentarios de transacciones
# ============================================================

import pandas as pd
import numpy as np
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import save_npz

# Descargar recursos NLTK
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# ============================================================
# CARGAR DATASET
# ============================================================

print("📦 Cargando creditcard.csv...")
df = pd.read_csv("data/creditcard.csv")
print(f"✅ Dataset: {df.shape[0]} transacciones")

# ============================================================
# GENERAR COMENTARIOS SINTÉTICOS REALISTAS
# ============================================================

print("\n📝 Generando comentarios de transacciones...")
np.random.seed(42)

comentarios_normal = [
    "pago supermercado compra semanal productos básicos",
    "retiro cajero automático banco central dinero efectivo",
    "transferencia pago arriendo mensual apartamento",
    "compra farmacia medicamentos receta médica",
    "pago servicio internet mensual proveedor hogar",
    "compra restaurante almuerzo ejecutivo centro comercial",
    "pago tarjeta crédito cuota mensual banco",
    "compra gasolina estación servicio combustible vehículo",
    "transferencia pago nómina empleados empresa",
    "compra ropa tienda descuento temporada",
]

comentarios_fraude = [
    "transferencia urgente cuenta extranjera desconocida madrugada",
    "retiro múltiple cajero límite máximo horario inusual",
    "compra electrónica costosa tienda online desconocida extranjero",
    "transferencia internacional cuenta nueva sin historial",
    "pago sospechoso monto elevado horario nocturno inusual",
    "retiro efectivo máximo cuenta nueva cliente reciente",
    "compra lujo tarjeta recién activada monto alto",
    "transferencia rápida múltiples cuentas diferentes países",
    "operación duplicada misma cuenta diferentes ciudades simultáneas",
    "compra en línea datos tarjeta nueva sin verificación",
]

comentarios = []
for i, row in df.iterrows():
    if row["Class"] == 1:
        comentarios.append(np.random.choice(comentarios_fraude))
    else:
        comentarios.append(np.random.choice(comentarios_normal))

df["comentario"] = comentarios
print(f"✅ Comentarios generados: {len(comentarios)}")

# ============================================================
# LIMPIEZA Y ANÁLISIS DE TEXTO
# ============================================================

print("\n🔍 Procesando texto con NLTK...")
stop_words = set(stopwords.words("spanish"))

def limpiar_texto(texto):
    tokens = word_tokenize(texto.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

df["texto_limpio"] = df["comentario"].apply(limpiar_texto)

# Análisis de N-gramas
print("\n📊 Análisis de N-gramas (bigramas y trigramas)...")
todos_tokens = " ".join(df["texto_limpio"]).split()

bigramas = list(ngrams(todos_tokens, 2))
trigramas = list(ngrams(todos_tokens, 3))

from collections import Counter
top_bigramas = Counter(bigramas).most_common(10)
top_trigramas = Counter(trigramas).most_common(10)

print("\nTop 10 Bigramas:")
for bg, count in top_bigramas:
    print(f"  {' '.join(bg)}: {count}")

print("\nTop 10 Trigramas:")
for tg, count in top_trigramas:
    print(f"  {' '.join(tg)}: {count}")

# Detección de entidades (NER) en muestra
print("\n🏷️ Análisis NER en muestra de comentarios...")
muestra = df["comentario"].head(10).tolist()
for texto in muestra[:3]:
    tokens = word_tokenize(texto)
    pos_tags = nltk.pos_tag(tokens)
    entidades = nltk.ne_chunk(pos_tags, binary=False)
    print(f"  Texto: {texto[:50]}...")
    for chunk in entidades:
        if hasattr(chunk, "label"):
            print(f"    Entidad [{chunk.label()}]: {' '.join(c[0] for c in chunk)}")

# ============================================================
# TF-IDF CON MATRIZ DISPERSA
# ============================================================

print("\n⚡ Aplicando TF-IDF con N-gramas (1,2,3)...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=5000,
    min_df=2,
    sublinear_tf=True
)

X_tfidf = tfidf.fit_transform(df["texto_limpio"])
print(f"✅ Matriz TF-IDF dispersa: {X_tfidf.shape}")
print(f"   Densidad: {X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.4%}")

# ============================================================
# TRUNCATED SVD - LSA (50 dimensiones)
# ============================================================

print("\n🔬 Aplicando Truncated SVD (LSA) - 50 dimensiones...")
svd = TruncatedSVD(n_components=50, random_state=42)
X_lsa = svd.fit_transform(X_tfidf)

varianza_explicada = svd.explained_variance_ratio_.sum()
print(f"✅ LSA completado: {X_lsa.shape}")
print(f"   Varianza explicada por 50 componentes: {varianza_explicada:.2%}")

# ============================================================
# GUARDAR RESULTADOS
# ============================================================

print("\n💾 Guardando resultados...")

# Agregar columnas LSA al dataframe
lsa_cols = [f"lsa_{i}" for i in range(50)]
df_lsa = pd.DataFrame(X_lsa, columns=lsa_cols)
df_final = pd.concat([df.reset_index(drop=True), df_lsa], axis=1)
df_final.to_csv("data/datos_con_texto.csv", index=False)

# Guardar modelos
with open("data/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("data/svd_lsa.pkl", "wb") as f:
    pickle.dump(svd, f)

save_npz("data/matriz_tfidf.npz", X_tfidf)

print("✅ datos_con_texto.csv guardado")
print("✅ tfidf_vectorizer.pkl guardado")
print("✅ svd_lsa.pkl guardado")
print("✅ matriz_tfidf.npz guardado")

print(f"\n{'='*55}")
print("📊 RESUMEN PASO 3")
print(f"{'='*55}")
print(f"   Transacciones procesadas: {len(df):,}")
print(f"   Vocabulario TF-IDF:       {X_tfidf.shape[1]:,} términos")
print(f"   Dimensiones LSA:          50")
print(f"   Varianza explicada:       {varianza_explicada:.2%}")
print(f"   Top bigrama:              {' '.join(top_bigramas[0][0])}")
print(f"{'='*55}")
print("\n🎉 Paso 3 completado exitosamente!")