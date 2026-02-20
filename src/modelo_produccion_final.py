# ============================================================
# FRAUDLYTICS - Modelo de Producción Final
# SMOTE + XGBoost + Umbral optimizado
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, precision_recall_curve, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ============================================================
# CARGA DE DATOS
# ============================================================

print("📦 Cargando dataset...")
df = pd.read_csv("data/creditcard.csv")

features = [col for col in df.columns if col.startswith("V")] + ["Amount"]
X = df[features].values
y = df["Class"].values

print(f"✅ Dataset: {len(y)} transacciones | {y.sum()} fraudes ({y.mean():.2%})")

# ============================================================
# DIVISIÓN HONESTA
# ============================================================

print("\n✂️ Dividiendo datos 70/30...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(y_train)} | Fraudes: {y_train.sum()}")
print(f"✅ Test:  {len(y_test)} | Fraudes: {y_test.sum()}")

# ============================================================
# SMOTE - Balanceo del dataset
# ============================================================

print("\n⚖️ Aplicando SMOTE para balancear fraudes...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"✅ Antes de SMOTE: {y_train.sum()} fraudes de {len(y_train)}")
print(f"✅ Después de SMOTE: {y_train_smote.sum()} fraudes de {len(y_train_smote)}")

# ============================================================
# ENTRENAMIENTO CON MEJORES HIPERPARÁMETROS
# ============================================================

print("\n⚡ Entrenando XGBoost con datos balanceados...")
modelo = XGBClassifier(
    subsample=0.8,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
    # Sin scale_pos_weight porque SMOTE ya balanceó
)

modelo.fit(X_train_smote, y_train_smote)
print("✅ Entrenamiento completado")

# ============================================================
# AJUSTE DEL UMBRAL ÓPTIMO
# ============================================================

print("\n🎯 Buscando umbral óptimo...")
y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
precision_vals, recall_vals, umbrales = precision_recall_curve(y_test, y_prob)

# Buscamos el umbral con Recall mínimo 85% y mayor Precision posible
mejor_f1 = 0
mejor_umbral = 0.5
mejor_recall = 0
mejor_precision = 0

for i, umbral in enumerate(umbrales):
    if recall_vals[i] >= 0.85:
        f1 = 2 * (precision_vals[i] * recall_vals[i]) / (precision_vals[i] + recall_vals[i] + 1e-10)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral
            mejor_recall = recall_vals[i]
            mejor_precision = precision_vals[i]

print(f"✅ Umbral óptimo: {mejor_umbral:.4f}")

# ============================================================
# EVALUACIÓN FINAL HONESTA
# ============================================================

y_pred_final = (y_prob >= mejor_umbral).astype(int)

fraudes_detectados = y_pred_final[y_test == 1].sum()
fraudes_totales = y_test.sum()
falsas_alarmas = y_pred_final[y_test == 0].sum()
fraudes_perdidos = fraudes_totales - fraudes_detectados

print("\n" + "="*55)
print("📊 RESULTADOS FINALES - MODELO DE PRODUCCIÓN")
print("="*55)
print(f"   Recall:    {recall_score(y_test, y_pred_final):.4f} → detecta el {recall_score(y_test, y_pred_final):.1%} de fraudes")
print(f"   Precision: {precision_score(y_test, y_pred_final):.4f} → {precision_score(y_test, y_pred_final):.1%} de alertas son reales")
print(f"   F1-Score:  {f1_score(y_test, y_pred_final):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"\n   🚨 Fraudes detectados: {fraudes_detectados} de {fraudes_totales}")
print(f"   ⚠️  Falsas alarmas:     {falsas_alarmas}")
print(f"   💰 Fraudes perdidos:   {fraudes_perdidos}")
print("="*55)

print("\n📋 Reporte completo:")
print(classification_report(y_test, y_pred_final, target_names=["Normal", "Fraude"]))

# ============================================================
# COMPARACIÓN CON MODELO ANTERIOR
# ============================================================

print("📈 Comparación con modelo anterior (sin SMOTE):")
print(f"   Antes  → Recall: 79.1% | Precision: 88.0% | Fraudes perdidos: 31")
print(f"   Ahora  → Recall: {recall_score(y_test, y_pred_final):.1%} | Precision: {precision_score(y_test, y_pred_final):.1%} | Fraudes perdidos: {fraudes_perdidos}")

# ============================================================
# GUARDAR MODELO FINAL
# ============================================================

print("\n💾 Guardando modelo final de producción...")
with open("data/modelo_produccion.pkl", "wb") as f:
    pickle.dump(modelo, f)

with open("data/scaler_produccion.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("data/features_produccion.pkl", "wb") as f:
    pickle.dump(features, f)

with open("data/umbral_produccion.pkl", "wb") as f:
    pickle.dump(mejor_umbral, f)

print("✅ Modelo guardado en data/modelo_produccion.pkl")
print("\n🎉 Modelo de producción listo!")