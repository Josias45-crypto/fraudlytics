# ============================================================
# FRAUDLYTICS - Entrenamiento con dataset IEEE-CIS
# Dataset más rico y realista para producción bancaria
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, precision_recall_curve
)
from xgboost import XGBClassifier

# ============================================================
# CARGA Y MERGE DE DATOS
# ============================================================

print("📦 Cargando datasets IEEE-CIS...")
train_transaction = pd.read_csv("data/train_transaction.csv")
train_identity = pd.read_csv("data/train_identity.csv")

print(f"✅ Transacciones: {train_transaction.shape}")
print(f"✅ Identidades:   {train_identity.shape}")

print("\n🔗 Haciendo merge de transacciones e identidades...")
df = train_transaction.merge(train_identity, on="TransactionID", how="left")
print(f"✅ Dataset combinado: {df.shape}")
print(f"📊 Fraudes: {df['isFraud'].sum()} ({df['isFraud'].mean():.2%})")

# ============================================================
# PREPARACIÓN DE FEATURES
# ============================================================

print("\n🔧 Preparando features...")

# Separar target
y = df["isFraud"].values

# Eliminar columnas que no son features
cols_eliminar = ["TransactionID", "isFraud", "TransactionDT"]
df = df.drop(cols_eliminar, axis=1)

# Encodear variables categóricas
categoricas = df.select_dtypes(include=["object"]).columns
print(f"   Variables categóricas encontradas: {len(categoricas)}")

le = LabelEncoder()
for col in categoricas:
    df[col] = df[col].fillna("desconocido")
    df[col] = le.fit_transform(df[col].astype(str))

# Rellenar nulos numéricos con la mediana
numericas = df.select_dtypes(include=[np.number]).columns
df[numericas] = df[numericas].fillna(df[numericas].median())

X = df.values
features = df.columns.tolist()

print(f"✅ Features totales: {len(features)}")
print(f"✅ Dataset listo: {X.shape}")

# ============================================================
# DIVISIÓN HONESTA 70/30
# ============================================================

print("\n✂️ Dividiendo datos 70/30...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(y_train)} | Fraudes: {y_train.sum()}")
print(f"✅ Test:  {len(y_test)} | Fraudes: {y_test.sum()}")

# ============================================================
# ENTRENAMIENTO XGBOOST
# ============================================================

print("\n⚡ Entrenando XGBoost con dataset IEEE-CIS...")
modelo = XGBClassifier(
    subsample=0.8,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

modelo.fit(X_train_scaled, y_train)
print("✅ Entrenamiento completado")

# ============================================================
# AJUSTE DEL UMBRAL ÓPTIMO
# ============================================================

print("\n🎯 Buscando umbral óptimo...")
y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
precision_vals, recall_vals, umbrales = precision_recall_curve(y_test, y_prob)

mejor_f1 = 0
mejor_umbral = 0.5

for i, umbral in enumerate(umbrales):
    if recall_vals[i] >= 0.85:
        f1 = 2 * (precision_vals[i] * recall_vals[i]) / (precision_vals[i] + recall_vals[i] + 1e-10)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

print(f"✅ Umbral óptimo: {mejor_umbral:.4f}")

# ============================================================
# EVALUACIÓN FINAL
# ============================================================

y_pred = (y_prob >= mejor_umbral).astype(int)

fraudes_detectados = y_pred[y_test == 1].sum()
fraudes_totales = y_test.sum()
falsas_alarmas = y_pred[y_test == 0].sum()

print("\n" + "="*55)
print("📊 RESULTADOS - MODELO IEEE-CIS")
print("="*55)
print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"\n   🚨 Fraudes detectados: {fraudes_detectados} de {fraudes_totales}")
print(f"   ⚠️  Falsas alarmas:     {falsas_alarmas}")
print(f"   💰 Fraudes perdidos:   {fraudes_totales - fraudes_detectados}")
print("="*55)

print("\n📋 Reporte completo:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"]))

# ============================================================
# GUARDAR MODELO
# ============================================================

print("\n💾 Guardando modelo IEEE-CIS...")
with open("data/modelo_produccion.pkl", "wb") as f:
    pickle.dump(modelo, f)

with open("data/scaler_produccion.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("data/features_produccion.pkl", "wb") as f:
    pickle.dump(features, f)

with open("data/umbral_produccion.pkl", "wb") as f:
    pickle.dump(mejor_umbral, f)

print("✅ Modelo guardado")
print("\n🎉 Modelo IEEE-CIS listo para producción!")