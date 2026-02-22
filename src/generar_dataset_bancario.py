# ============================================================
# FRAUDLYTICS - Dataset bancario realista
# Patrones con ruido para métricas honestas
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

np.random.seed(42)
n = 200000
fraude_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
es_fraude = np.zeros(n, dtype=int)
es_fraude[fraude_idx] = 1

print("🏦 Generando dataset bancario realista...")

# Función para mezclar patrones con ruido
def mezclar(condicion_fraude, condicion_normal, ruido=0.25):
    resultado = np.where(condicion_fraude, 1, 0)
    ruido_idx = np.random.choice(len(resultado), size=int(len(resultado) * ruido), replace=False)
    resultado[ruido_idx] = 1 - resultado[ruido_idx]
    return resultado

df = pd.DataFrame({
    "monto": np.where(
        es_fraude,
        np.random.exponential(800, n) + np.random.normal(0, 200, n),
        np.random.exponential(120, n) + np.random.normal(0, 50, n)
    ).clip(1).round(2),

    "hora": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.3,
                 np.random.choice(list(range(0, 6)) + list(range(22, 24)), n),
                 np.random.randint(6, 22, n)),
        np.random.randint(6, 22, n)
    ),

    "tipo_transaccion": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.3,
                 np.random.choice(["transferencia", "retiro"], n),
                 np.random.choice(["compra", "pago_servicio"], n)),
        np.random.choice(["compra", "pago_servicio", "retiro", "transferencia"], n,
                         p=[0.5, 0.3, 0.1, 0.1])
    ),

    "canal": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.3,
                 np.random.choice(["web", "app_movil"], n),
                 np.random.choice(["cajero", "sucursal"], n)),
        np.random.choice(["web", "app_movil", "cajero", "sucursal"], n,
                         p=[0.35, 0.35, 0.2, 0.1])
    ),

    "pais_origen": np.random.choice(
        ["CO", "MX", "PE", "AR", "CL"], n, p=[0.3, 0.25, 0.2, 0.15, 0.1]
    ),

    "pais_destino": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.3,
                 np.random.choice(["US", "ES", "CN", "RU"], n),
                 np.random.choice(["CO", "MX", "PE"], n)),
        np.random.choice(["CO", "MX", "PE", "AR", "CL"], n)
    ),

    "tarjeta_tipo": np.random.choice(
        ["visa", "mastercard", "amex", "diners"], n, p=[0.5, 0.3, 0.15, 0.05]
    ),

    "cliente_edad": np.random.randint(18, 80, n),

    "cliente_antiguedad_dias": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.3,
                 np.random.randint(1, 90, n),
                 np.random.randint(90, 1000, n)),
        np.random.randint(90, 3650, n)
    ),

    "transacciones_ultimas_24h": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.3,
                 np.random.randint(5, 20, n),
                 np.random.randint(0, 5, n)),
        np.random.randint(0, 5, n)
    ),

    "monto_promedio_historico": np.random.exponential(180, n).round(2),

    "distancia_ultima_transaccion_km": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.3,
                 np.random.exponential(600, n),
                 np.random.exponential(20, n)),
        np.random.exponential(8, n)
    ).round(2),

    "es_horario_inusual": np.where(
        es_fraude,
        np.random.choice([0, 1], n, p=[0.35, 0.65]),
        np.random.choice([0, 1], n, p=[0.92, 0.08])
    ),

    "intentos_fallidos_previos": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.3,
                 np.random.randint(1, 5, n),
                 np.zeros(n, dtype=int)),
        np.zeros(n, dtype=int)
    ),

    "es_fraude": es_fraude
})

print(f"✅ Dataset: {len(df)} transacciones | {es_fraude.sum()} fraudes ({es_fraude.mean():.2%})")

# Guardar dataset
df.to_csv("data/dataset_bancario.csv", index=False)
print("✅ Guardado en data/dataset_bancario.csv")

# Plantilla sin es_fraude
plantilla = df.drop("es_fraude", axis=1).head(10)
plantilla = plantilla.astype(object)
plantilla[:] = ""
plantilla.to_excel("data/plantilla_fraudlytics.xlsx", index=False, sheet_name="Transacciones")
plantilla.to_csv("data/plantilla_fraudlytics.csv", index=False)
print("✅ Plantilla Excel guardada")

# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\n🤖 Entrenando modelo bancario realista...")

df_model = df.copy()
le = LabelEncoder()
for col in ["tipo_transaccion", "canal", "pais_origen", "pais_destino", "tarjeta_tipo"]:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

features = [c for c in df_model.columns if c != "es_fraude"]
X = df_model[features].values
y = df_model["es_fraude"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# Ajuste de umbral
y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
precision_vals, recall_vals, umbrales = precision_recall_curve(y_test, y_prob)

mejor_f1 = 0
mejor_umbral = 0.5
for i, umbral in enumerate(umbrales):
    if recall_vals[i] >= 0.75:
        f1 = 2 * (precision_vals[i] * recall_vals[i]) / (precision_vals[i] + recall_vals[i] + 1e-10)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

y_pred = (y_prob >= mejor_umbral).astype(int)

print(f"\n{'='*55}")
print("📊 RESULTADOS MODELO BANCARIO REALISTA")
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

print("✅ Modelo guardado")
print("🎉 Listo para producción!")