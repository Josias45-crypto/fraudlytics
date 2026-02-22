# ============================================================
# FRAUDLYTICS - Generador de plantilla Excel
# ============================================================

import pandas as pd
import pickle
import numpy as np

print("📦 Cargando features del modelo...")
with open("data/features_produccion.pkl", "rb") as f:
    features = pickle.load(f)

print(f"✅ El modelo espera {len(features)} columnas")

# Crear plantilla con 5 filas de ejemplo
plantilla = pd.DataFrame(columns=features)
for i in range(5):
    fila = {col: 0.0 for col in features}
    plantilla = pd.concat([plantilla, pd.DataFrame([fila])], ignore_index=True)

# Guardar como Excel con formato
with pd.ExcelWriter("data/plantilla_fraudlytics.xlsx", engine="openpyxl") as writer:
    plantilla.to_excel(writer, index=False, sheet_name="Transacciones")
    
    # Formato visual
    workbook = writer.book
    worksheet = writer.sheets["Transacciones"]
    
    # Ancho de columnas
    for col in worksheet.columns:
        worksheet.column_dimensions[col[0].column_letter].width = 15

print("✅ Plantilla Excel guardada en data/plantilla_fraudlytics.xlsx")
print(f"✅ Columnas: {len(features)}")
print("\n🎉 Comparte este Excel con tus usuarios!")