# ============================================================
# FRAUDLYTICS - Optimización del modelo para producción
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, precision_recall_curve, make_scorer
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Dataset listo: {len(y)} transacciones | {y.sum()} fraudes")

# ============================================================
# BÚSQUEDA DE HIPERPARÁMETROS
# ============================================================

print("\n🔍 Buscando mejores hiperparámetros para XGBoost...")
print("   Esto puede tardar varios minutos...")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "scale_pos_weight": [
        len(y[y==0])/len(y[y==1]),
        len(y[y==0])/len(y[y==1]) * 1.5,
        len(y[y==0])/len(y[y==1]) * 2
    ]
}

modelo_base = XGBClassifier(random_state=42, eval_metric="logloss")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Usamos F1 como métrica de búsqueda
busqueda = RandomizedSearchCV(
    modelo_base,
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1",
    cv=skf,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

busqueda.fit(X_scaled, y)

print(f"\n✅ Mejores hiperparámetros encontrados:")
for param, valor in busqueda.best_params_.items():
    print(f"   {param}: {valor}")

# ============================================================
# ENTRENAMIENTO CON MEJORES PARÁMETROS
# ============================================================

print("\n⚡ Entrenando modelo optimizado...")
modelo_optimizado = busqueda.best_estimator_
modelo_optimizado.fit(X_scaled, y)

y_prob = modelo_optimizado.predict_proba(X_scaled)[:, 1]

# ============================================================
# AJUSTE DEL UMBRAL DE DECISIÓN
# ============================================================

print("\n🎯 Ajustando umbral de decisión...")
precision_vals, recall_vals, umbrales = precision_recall_curve(y, y_prob)

# Buscamos el umbral que maximiza F1 con Recall mínimo del 85%
mejor_f1 = 0
mejor_umbral = 0.5

for i, umbral in enumerate(umbrales):
    if recall_vals[i] >= 0.85:
        f1 = 2 * (precision_vals[i] * recall_vals[i]) / (precision_vals[i] + recall_vals[i] + 1e-10)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

print(f"✅ Umbral óptimo encontrado: {mejor_umbral:.4f}")

# Evaluación final con umbral optimizado
y_pred_final = (y_prob >= mejor_umbral).astype(int)

print(f"\n📊 Resultados finales del modelo optimizado:")
print(f"   Recall:    {recall_score(y, y_pred_final):.4f}")
print(f"   Precision: {precision_score(y, y_pred_final):.4f}")
print(f"   F1-Score:  {f1_score(y, y_pred_final):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y, y_prob):.4f}")

fraudes_detectados = y_pred_final[y == 1].sum()
fraudes_totales = y.sum()
print(f"\n🚨 Fraudes detectados: {fraudes_detectados} de {fraudes_totales} ({fraudes_detectados/fraudes_totales:.1%})")

falsas_alarmas = y_pred_final[y == 0].sum()
print(f"⚠️  Falsas alarmas:     {falsas_alarmas} de {(y==0).sum()} ({falsas_alarmas/(y==0).sum():.2%})")

# ============================================================
# GUARDAR MODELO OPTIMIZADO
# ============================================================

print("\n💾 Guardando modelo optimizado...")
with open("data/modelo_optimizado.pkl", "wb") as f:
    pickle.dump(modelo_optimizado, f)

with open("data/scaler_final.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("data/features_finales.pkl", "wb") as f:
    pickle.dump(features, f)

with open("data/umbral_optimo.pkl", "wb") as f:
    pickle.dump(mejor_umbral, f)

print("✅ Todos los archivos guardados")
print(f"\n🎉 Modelo listo para producción!")