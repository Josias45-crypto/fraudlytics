# ============================================================
# FRAUDLYTICS - Ajuste de umbral óptimo
# Encontrar el mejor balance Precision/Recall
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score
)

# ============================================================
# CARGA DE DATOS
# ============================================================

print("📦 Cargando datasets...")
train_transaction = pd.read_csv("data/train_transaction.csv")
train_identity = pd.read_csv("data/train_identity.csv")
df = train_transaction.merge(train_identity, on="TransactionID", how="left")

y = df["isFraud"].values
df = df.drop(["TransactionID", "isFraud", "TransactionDT"], axis=1)

categoricas = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categoricas:
    df[col] = df[col].fillna("desconocido")
    df[col] = le.fit_transform(df[col].astype(str))

numericas = df.select_dtypes(include=[np.number]).columns
df[numericas] = df[numericas].fillna(df[numericas].median())

X = df.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# CARGAR MODELO ENTRENADO
# ============================================================

print("🔃 Cargando modelo entrenado...")
with open("data/modelo_produccion.pkl", "rb") as f:
    modelo = pickle.load(f)

y_prob = modelo.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# PROBAR DIFERENTES UMBRALES
# ============================================================

print("\n🎯 Probando umbrales para encontrar el mejor balance...\n")
print(f"{'Umbral':<10} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Fraudes perdidos':<18} {'Falsas alarmas'}")
print("-" * 75)

resultados = []
for umbral in np.arange(0.1, 0.95, 0.05):
    y_pred = (y_prob >= umbral).astype(int)
    
    if y_pred.sum() == 0:
        continue
        
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    perdidos = y_test.sum() - y_pred[y_test == 1].sum()
    falsas = y_pred[y_test == 0].sum()
    
    resultados.append({
        "umbral": umbral,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "perdidos": perdidos,
        "falsas": falsas
    })
    
    print(f"{umbral:<10.2f} {recall:<10.4f} {precision:<12.4f} {f1:<10.4f} {perdidos:<18} {falsas}")

# ============================================================
# ELEGIR EL MEJOR UMBRAL PARA PRODUCCIÓN BANCARIA
# ============================================================

# Criterio: Recall mínimo 75% y máxima Precision posible
print("\n" + "="*55)
print("🏆 MEJOR UMBRAL PARA PRODUCCIÓN BANCARIA")
print("="*55)

candidatos = [r for r in resultados if r["recall"] >= 0.75]

if candidatos:
    mejor = max(candidatos, key=lambda x: x["precision"])
    print(f"\n   Umbral:           {mejor['umbral']:.2f}")
    print(f"   Recall:           {mejor['recall']:.4f} → detecta {mejor['recall']:.1%} de fraudes")
    print(f"   Precision:        {mejor['precision']:.4f} → {mejor['precision']:.1%} de alertas son reales")
    print(f"   F1-Score:         {mejor['f1']:.4f}")
    print(f"   Fraudes perdidos: {mejor['perdidos']}")
    print(f"   Falsas alarmas:   {mejor['falsas']}")

    # Guardar umbral óptimo
    with open("data/umbral_produccion.pkl", "wb") as f:
        pickle.dump(mejor["umbral"], f)

    print(f"\n✅ Umbral {mejor['umbral']:.2f} guardado como umbral de producción")
    print("\n🎉 Listo! Este es el balance óptimo para un banco real.")
else:
    print("⚠️ No se encontró umbral con Recall >= 75%")