# ============================================================
# FRAUDLYTICS - Entrenamiento del modelo final
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

# ============================================================
# CARGA DE DATOS
# ============================================================

print("📦 Cargando dataset...")
df = pd.read_csv("data/creditcard.csv")
print(f"✅ Dataset: {df.shape[0]} filas")
print(f"📊 Fraudes: {df['Class'].sum()} ({df['Class'].mean():.2%})")

features = [col for col in df.columns if col.startswith("V")] + ["Amount"]
X = df[features].values
y = df["Class"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================

print("\n🔬 Comparando algoritmos con StratifiedKFold...")

modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        scale_pos_weight=len(y[y==0])/len(y[y==1]),
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

for nombre, modelo in modelos.items():
    print(f"\n⏳ Evaluando {nombre}...")
    f1 = cross_val_score(modelo, X_scaled, y, cv=skf, scoring="f1", n_jobs=-1)
    recall = cross_val_score(modelo, X_scaled, y, cv=skf, scoring="recall", n_jobs=-1)
    roc = cross_val_score(modelo, X_scaled, y, cv=skf, scoring="roc_auc", n_jobs=-1)

    resultados[nombre] = {
        "F1": f1.mean(),
        "Recall": recall.mean(),
        "ROC-AUC": roc.mean()
    }

    print(f"   F1:      {f1.mean():.4f} ± {f1.std():.4f}")
    print(f"   Recall:  {recall.mean():.4f} ± {recall.std():.4f}")
    print(f"   ROC-AUC: {roc.mean():.4f} ± {roc.std():.4f}")

# ============================================================
# ELEGIR EL MEJOR MODELO
# ============================================================

mejor = max(resultados, key=lambda x: resultados[x]["Recall"])
print(f"\n🏆 Mejor modelo por Recall: {mejor}")

# ============================================================
# ENTRENAR EL MODELO FINAL
# ============================================================

print(f"\n⚡ Entrenando {mejor} con todos los datos...")
modelo_final = modelos[mejor]
modelo_final.fit(X_scaled, y)

y_pred = modelo_final.predict(X_scaled)
y_prob = modelo_final.predict_proba(X_scaled)[:, 1]

print(f"✅ F1-Score final:  {f1_score(y, y_pred):.4f}")
print(f"✅ Recall final:    {recall_score(y, y_pred):.4f}")
print(f"✅ Precision final: {precision_score(y, y_pred):.4f}")
print(f"✅ ROC-AUC final:   {roc_auc_score(y, y_prob):.4f}")

# ============================================================
# GUARDAR MODELO Y SCALER
# ============================================================

print("\n💾 Guardando modelo y scaler...")
with open("data/modelo_final.pkl", "wb") as f:
    pickle.dump(modelo_final, f)

with open("data/scaler_final.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("data/features_finales.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Guardado en data/modelo_final.pkl")
print("✅ Guardado en data/scaler_final.pkl")
print(f"\n🎉 Modelo {mejor} listo para producción!")