# ============================================================
# FRAUDLYTICS - PASO 1: Preparación de Datos
# Pandas, NumPy y SciPy
# ============================================================

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import ks_2samp

# ============================================================
# BLOQUE A - PANDAS
# ============================================================

print("📦 Cargando dataset...")
df = pd.read_csv("data/creditcard.csv")
print(f"✅ Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
print(df.head())

print("\n🔗 Creando tabla de usuarios y haciendo merge...")
usuarios = pd.DataFrame({
    "user_id": range(len(df)),
    "region": np.random.choice(["Norte", "Sur", "Este", "Oeste"], size=len(df)),
    "tipo_cuenta": np.random.choice(["premium", "basica"], size=len(df))
})
df["user_id"] = range(len(df))
df = df.merge(usuarios, on="user_id", how="left")
print(f"✅ Merge completado. Nuevas columnas: region, tipo_cuenta")

print("\n⏱️ Calculando gasto promedio últimas 48 horas...")
df = df.sort_values("Time")
df["gasto_promedio_48h"] = df["Amount"].rolling(window=172800, min_periods=1).mean()
print("✅ Window function aplicada")

# ============================================================
# BLOQUE B - NUMPY
# ============================================================

print("\n🔢 Aplicando Target Encoding...")
categorias = df["region"].values
target = df["Class"].values

encoding = {}
for cat in np.unique(categorias):
    mask = categorias == cat
    encoding[cat] = target[mask].mean()

df["region_encoded"] = np.vectorize(encoding.get)(df["region"])
print(f"✅ Target Encoding aplicado: {encoding}")

# ============================================================
# BLOQUE C - SCIPY
# ============================================================

print("\n📉 Aplicando filtro Savitzky-Golay...")
df["Amount_suavizado"] = savgol_filter(df["Amount"].values, window_length=51, polyorder=3)
print("✅ Columna Amount_suavizado creada")

print("\n📊 Test de Kolmogorov-Smirnov...")
normales = df[df["Class"] == 0]["Amount"].values
sospechosas = df[df["Class"] == 1]["Amount"].values

stat, p_value = ks_2samp(normales, sospechosas)
print(f"✅ Estadístico KS: {stat:.4f}")
print(f"✅ P-value: {p_value:.6f}")

if p_value < 0.05:
    print("🚨 Las distribuciones son DIFERENTES. El monto SÍ es indicador de fraude.")
else:
    print("✅ Las distribuciones son similares.")

print("\n💾 Guardando datos procesados...")
df.to_csv("data/datos_procesados.csv", index=False)
print("✅ Guardado en data/datos_procesados.csv")