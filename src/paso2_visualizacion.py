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

# ============================================================
# BLOQUE D - SUBPLOTS 2x2 COMPLETOS
# Superficie de decisión + Curva de aprendizaje
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

print("\n📊 Generando subplots 2x2 con superficie de decisión y curva de aprendizaje...")

# Reducimos a 2D con PCA para poder graficar la superficie de decisión
X_features = df_muestra[[col for col in df_muestra.columns if col.startswith("V")]].values
y_labels = df_muestra["Class"].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.fit_transform(X_features))

# Entrenamos modelo para superficie de decisión
modelo_decision = LogisticRegression(max_iter=1000, class_weight="balanced")
modelo_decision.fit(X_pca, y_labels)

# Entrenamos red neuronal para curva de aprendizaje
modelo_nn = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=50,
    random_state=42,
    warm_start=True
)

loss_curve = []
for i in range(1, 51):
    modelo_nn.max_iter = i
    modelo_nn.fit(X_pca, y_labels)
    loss_curve.append(modelo_nn.loss_)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Fraudlytics - Análisis Avanzado", fontsize=16)

# Gráfica 1: t-SNE scatterplot
axes[0, 0].scatter(
    X_tsne[y == 0, 0], X_tsne[y == 0, 1],
    c="steelblue", alpha=0.4, s=10, label="Normal"
)
axes[0, 0].scatter(
    X_tsne[y == 1, 0], X_tsne[y == 1, 1],
    c="red", alpha=0.8, s=30, label="Fraude"
)
axes[0, 0].set_title("t-SNE: Normales vs Fraudes")
axes[0, 0].legend()

# Gráfica 2: Distribución de montos suavizados
axes[0, 1].plot(df["Amount"].values[:500], color="gray", alpha=0.5, label="Original")
axes[0, 1].plot(df["Amount_suavizado"].values[:500], color="blue", lw=2, label="Suavizado")
axes[0, 1].set_title("Filtro Savitzky-Golay sobre Montos")
axes[0, 1].legend()

# Gráfica 3: Superficie de decisión
h = 0.5
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = modelo_decision.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[1, 0].contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
axes[1, 0].scatter(X_pca[y_labels==0, 0], X_pca[y_labels==0, 1], c="steelblue", alpha=0.4, s=10, label="Normal")
axes[1, 0].scatter(X_pca[y_labels==1, 0], X_pca[y_labels==1, 1], c="red", alpha=0.8, s=30, label="Fraude")
axes[1, 0].set_title("Superficie de Decisión del Modelo")
axes[1, 0].legend()

# Gráfica 4: Curva de aprendizaje Loss vs Epochs
axes[1, 1].plot(range(1, 51), loss_curve, color="purple", lw=2)
axes[1, 1].set_title("Curva de Aprendizaje (Loss vs Epochs)")
axes[1, 1].set_xlabel("Epochs")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/subplots_avanzados.png", dpi=150)
plt.show()
print("✅ Subplots 2x2 avanzados guardados")

print("\n🎉 Paso 2 completado exitosamente!")