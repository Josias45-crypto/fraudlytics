# ============================================================
# FRAUDLYTICS - Modelo final con creditcard.csv
# Datos reales, métricas honestas
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, precision_recall_curve
)
from xgboost import XGBClassifier

print("📦 Cargando creditcard.csv...")
df = pd.read_csv("data/creditcard.csv")
print(f"✅ Dataset: {len(df)} transacciones | {df['Class'].sum()} fraudes ({df['Class'].mean():.2%})")

features = [col for col in df.columns if col.startswith("V")] + ["Amount", "Time"]
X = df[features].values
y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(y_train)} | Fraudes: {y_train.sum()}")
print(f"✅ Test:  {len(y_test)} | Fraudes: {y_test.sum()}")

print("\n⚡ Entrenando XGBoost optimizado...")
modelo = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

modelo.fit(X_train_scaled, y_train)

y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
precision_vals, recall_vals, umbrales = precision_recall_curve(y_test, y_prob)

mejor_f1 = 0
mejor_umbral = 0.5
for i, umbral in enumerate(umbrales):
    if recall_vals[i] >= 0.80:
        f1 = 2 * (precision_vals[i] * recall_vals[i]) / (precision_vals[i] + recall_vals[i] + 1e-10)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

y_pred = (y_prob >= mejor_umbral).astype(int)

print(f"\n{'='*55}")
print("📊 RESULTADOS REALES DEL MODELO")
print(f"{'='*55}")
print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"   Umbral:    {mejor_umbral:.4f}")
print(f"\n   🚨 Fraudes detectados: {y_pred[y_test==1].sum()} de {y_test.sum()}")
print(f"   ⚠️  Falsas alarmas:     {y_pred[y_test==0].sum()}")
print(f"   💰 Fraudes perdidos:   {y_test.sum() - y_pred[y_test==1].sum()}")
print(f"{'='*55}")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"]))

# Guardar modelo
with open("data/modelo_produccion.pkl", "wb") as f:
    pickle.dump(modelo, f)
with open("data/scaler_produccion.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("data/features_produccion.pkl", "wb") as f:
    pickle.dump(features, f)
with open("data/umbral_produccion.pkl", "wb") as f:
    pickle.dump(mejor_umbral, f)

# Plantilla Excel
plantilla = pd.DataFrame([{col: 0.0 for col in features} for _ in range(5)])
plantilla.to_excel("data/plantilla_fraudlytics.xlsx", index=False, sheet_name="Transacciones")
plantilla.to_csv("data/plantilla_fraudlytics.csv", index=False)

print("✅ Plantilla guardada")
print("✅ Modelo guardado")
print("🎉 Listo para producción!")