# ============================================================
# FRAUDLYTICS - PASO 2: Visualización de Alta Dimensionalidad
# Matplotlib y Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ============================================================
# CARGA DE DATOS PROCESADOS
# ============================================================

print("📦 Cargando datos procesados...")
df = pd.read_csv("data/datos_procesados.csv")
print(f"✅ Datos cargados: {df.shape[0]} filas")

# Tomamos una muestra para que t-SNE no tarde demasiado
df_muestra = df.sample(n=3000, random_state=42)

# ============================================================
# BLOQUE A - t-SNE: Reducción de dimensionalidad
# ============================================================

print("\n🔍 Aplicando t-SNE para reducir dimensionalidad...")
features = [col for col in df_muestra.columns if col.startswith("V")]
X = df_muestra[features].values
y = df_muestra["Class"].values
amount = df_muestra["Amount"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
print("✅ t-SNE aplicado correctamente")

# ============================================================
# BLOQUE B - SEABORN: Scatterplot con color y tamaño
# ============================================================

print("\n🎨 Generando scatterplot con Seaborn...")
df_tsne = pd.DataFrame({
    "componente_1": X_tsne[:, 0],
    "componente_2": X_tsne[:, 1],
    "Class": y,
    "Amount": amount
})

plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df_tsne,
    x="componente_1",
    y="componente_2",
    hue="Class",
    size="Amount",
    sizes=(20, 200),
    palette={0: "steelblue", 1: "red"},
    alpha=0.6
)
plt.title("t-SNE: Transacciones Normales vs Fraudulentas")
plt.legend(title="Clase (0=Normal, 1=Fraude)")
plt.savefig("data/tsne_scatterplot.png", dpi=150)
plt.show()
print("✅ Scatterplot guardado")

# ============================================================
# BLOQUE C - MATPLOTLIB: Subplots 2x2
# ============================================================

print("\n📊 Generando subplots 2x2 con Matplotlib...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fraudlytics - Panel de Análisis", fontsize=16)

# Gráfica 1: Distribución de montos normales vs fraude
axes[0, 0].hist(df[df["Class"] == 0]["Amount"], bins=50, color="steelblue", alpha=0.7, label="Normal")
axes[0, 0].hist(df[df["Class"] == 1]["Amount"], bins=50, color="red", alpha=0.7, label="Fraude")
axes[0, 0].set_title("Distribución de Montos")
axes[0, 0].legend()

# Gráfica 2: Monto suavizado vs original
axes[0, 1].plot(df["Amount"].values[:500], color="gray", alpha=0.5, label="Original")
axes[0, 1].plot(df["Amount_suavizado"].values[:500], color="blue", label="Suavizado")
axes[0, 1].set_title("Filtro Savitzky-Golay sobre Montos")
axes[0, 1].legend()

# Gráfica 3: Conteo de fraudes vs normales
clase_counts = df["Class"].value_counts()
axes[1, 0].bar(["Normal", "Fraude"], clase_counts.values, color=["steelblue", "red"])
axes[1, 0].set_title("Conteo de Transacciones")

# Gráfica 4: Gasto promedio 48h
axes[1, 1].plot(df["gasto_promedio_48h"].values[:500], color="green")
axes[1, 1].set_title("Gasto Promedio Últimas 48h")

plt.tight_layout()
plt.savefig("data/panel_analisis.png", dpi=150)
plt.show()
print("✅ Panel de subplots guardado")

print("\n🎉 Paso 2 completado exitosamente!")