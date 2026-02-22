# ============================================================
# FRAUDLYTICS - Generador de plantilla Excel
# Basada en el modelo entrenado con creditcard.csv
# ============================================================

import pandas as pd
import pickle

print("📦 Cargando features del modelo...")
with open("data/features_produccion.pkl", "rb") as f:
    features = pickle.load(f)

print(f"✅ El modelo usa {len(features)} columnas")

# Plantilla con 5 filas de ejemplo
plantilla = pd.DataFrame([{col: 0.0 for col in features} for _ in range(5)])

# Guardar Excel y CSV
plantilla.to_excel("data/plantilla_fraudlytics.xlsx", index=False, sheet_name="Transacciones")
plantilla.to_csv("data/plantilla_fraudlytics.csv", index=False)

print("✅ Plantilla Excel guardada en data/plantilla_fraudlytics.xlsx")
print("✅ Plantilla CSV guardada en data/plantilla_fraudlytics.csv")
print(f"\n📋 Columnas incluidas: {features}")
print("\n🎉 Comparte la plantilla Excel con tus usuarios!")