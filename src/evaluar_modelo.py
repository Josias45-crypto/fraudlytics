# ============================================================
# FRAUDLYTICS - Evaluación honesta del modelo
# Train/Test split real
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# ============================================================
# CARGA DE DATOS
# ============================================================

print("📦 Cargando dataset...")
df = pd.read_csv("data/creditcard.csv")

features = [col for col in df.columns if col.startswith("V")] + ["Amount"]
X = df[features].values
y = df["Class"].values

# ============================================================
# DIVISIÓN HONESTA: 70% entrenamiento / 30% prueba
# ============================================================

print("\n✂️ Dividiendo datos: 70% entrenamiento | 30% prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y  # mantiene el mismo % de fraudes en ambos grupos
)

print(f"✅ Entrenamiento: {len(y_train)} transacciones | {y_train.sum()} fraudes")
print(f"✅ Prueba:        {len(y_test)} transacciones | {y_test.sum()} fraudes")

# Escalado - IMPORTANTE: fit solo con train, transform a ambos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # solo transform, no fit

# ============================================================
# CARGAR MEJORES HIPERPARÁMETROS DEL MODELO OPTIMIZADO
# ============================================================

print("\n⚡ Entrenando XGBoost optimizado SOLO con datos de entrenamiento...")

modelo = XGBClassifier(
    subsample=0.8,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)

modelo.fit(X_train_scaled, y_train)
print("✅ Entrenamiento completado")

# ============================================================
# EVALUACIÓN CON DATOS QUE NUNCA VIO
# ============================================================

print("\n🧪 Evaluando con datos de prueba (nunca vistos)...")
y_pred = modelo.predict(X_test_scaled)
y_prob = modelo.predict_proba(X_test_scaled)[:, 1]

# Cargar umbral óptimo
with open("data/umbral_optimo.pkl", "rb") as f:
    umbral = pickle.load(f)

print(f"🎯 Usando umbral óptimo: {umbral:.4f}")
y_pred_umbral = (y_prob >= umbral).astype(int)

print("\n" + "="*50)
print("📊 RESULTADOS HONESTOS DEL MODELO")
print("="*50)

for nombre, pred in [("Umbral estándar (0.5)", y_pred), ("Umbral optimizado", y_pred_umbral)]:
    print(f"\n🔹 {nombre}:")
    print(f"   Recall:    {recall_score(y_test, pred):.4f} → detecta el {recall_score(y_test, pred):.1%} de fraudes reales")
    print(f"   Precision: {precision_score(y_test, pred):.4f} → {precision_score(y_test, pred):.1%} de alertas son fraudes reales")
    print(f"   F1-Score:  {f1_score(y_test, pred):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

    fraudes_detectados = pred[y_test == 1].sum()
    fraudes_totales = y_test.sum()
    falsas_alarmas = pred[y_test == 0].sum()

    print(f"\n   🚨 Fraudes detectados: {fraudes_detectados} de {fraudes_totales}")
    print(f"   ⚠️  Falsas alarmas:     {falsas_alarmas}")
    print(f"   💰 Fraudes perdidos:   {fraudes_totales - fraudes_detectados}")

print("\n" + "="*50)
print("📋 REPORTE DETALLADO (umbral optimizado)")
print("="*50)
print(classification_report(y_test, y_pred_umbral, target_names=["Normal", "Fraude"]))

# ============================================================
# GUARDAR MODELO FINAL HONESTO
# ============================================================

print("\n💾 Guardando modelo final honesto...")
with open("data/modelo_produccion.pkl", "wb") as f:
    pickle.dump(modelo, f)

with open("data/scaler_produccion.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("data/features_produccion.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Modelo de producción guardado")
print("\n🎉 Este es el rendimiento real que tendrá con datos nuevos!")