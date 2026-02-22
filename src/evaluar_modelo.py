# ============================================================
# FRAUDLYTICS - Modelo final con 14 columnas bancarias
# Datos generados con distribuciones reales del creditcard.csv
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

print("📦 Cargando distribuciones reales del creditcard.csv...")
df_real = pd.read_csv("data/creditcard.csv")

monto_fraude_mean = df_real[df_real["Class"]==1]["Amount"].mean()
monto_fraude_std = df_real[df_real["Class"]==1]["Amount"].std()
monto_normal_mean = df_real[df_real["Class"]==0]["Amount"].mean()
monto_normal_std = df_real[df_real["Class"]==0]["Amount"].std()

print(f"✅ Monto promedio fraude real: ${monto_fraude_mean:.2f}")
print(f"✅ Monto promedio normal real: ${monto_normal_mean:.2f}")

print("\n🏦 Generando dataset bancario con distribuciones reales...")
np.random.seed(42)
n = 200000
fraude_idx = np.random.choice(n, size=int(n * 0.0017), replace=False)
es_fraude = np.zeros(n, dtype=int)
es_fraude[fraude_idx] = 1

df = pd.DataFrame({
    "monto": np.where(
        es_fraude,
        np.abs(np.random.normal(monto_fraude_mean, monto_fraude_std, n)),
        np.abs(np.random.normal(monto_normal_mean, monto_normal_std, n))
    ).round(2),
    "hora": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.4,
                 np.random.choice(list(range(0, 6)) + list(range(22, 24)), n),
                 np.random.randint(6, 22, n)),
        np.random.randint(6, 22, n)
    ),
    "tipo_transaccion": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.4,
                 np.random.choice(["transferencia", "retiro"], n),
                 np.random.choice(["compra", "pago_servicio"], n)),
        np.random.choice(["compra", "pago_servicio", "retiro", "transferencia"], n,
                         p=[0.5, 0.3, 0.1, 0.1])
    ),
    "canal": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.4,
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
        np.where(np.random.random(n) > 0.4,
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
        np.where(np.random.random(n) > 0.4,
                 np.random.randint(1, 90, n),
                 np.random.randint(90, 1000, n)),
        np.random.randint(90, 3650, n)
    ),
    "transacciones_ultimas_24h": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.4,
                 np.random.randint(5, 20, n),
                 np.random.randint(0, 5, n)),
        np.random.randint(0, 4, n)
    ),
    "monto_promedio_historico": np.abs(
        np.random.normal(monto_normal_mean, monto_normal_std, n)
    ).round(2),
    "distancia_ultima_transaccion_km": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.4,
                 np.random.exponential(600, n),
                 np.random.exponential(20, n)),
        np.random.exponential(8, n)
    ).round(2),
    "es_horario_inusual": np.where(
        es_fraude,
        np.random.choice([0, 1], n, p=[0.4, 0.6]),
        np.random.choice([0, 1], n, p=[0.92, 0.08])
    ),
    "intentos_fallidos_previos": np.where(
        es_fraude,
        np.where(np.random.random(n) > 0.4,
                 np.random.randint(1, 5, n),
                 np.zeros(n, dtype=int)),
        np.zeros(n, dtype=int)
    ),
    "es_fraude": es_fraude
})

print(f"✅ Dataset: {len(df)} transacciones | {es_fraude.sum()} fraudes ({es_fraude.mean():.2%})")

print("\n🤖 Entrenando modelo...")
df_model = df.copy()
le = LabelEncoder()
categoricas = ["tipo_transaccion", "canal", "pais_origen", "pais_destino", "tarjeta_tipo"]
for col in categoricas:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Guardar clases de cada columna categórica
encoders = {}
le2 = LabelEncoder()
for col in categoricas:
    encoders[col] = LabelEncoder()
    df_model[col] = encoders[col].fit_transform(df[col].astype(str))

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
print("📊 RESULTADOS FINALES")
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

# Guardar modelo y todo lo necesario
with open("data/modelo_produccion.pkl", "wb") as f:
    pickle.dump(modelo, f)
with open("data/scaler_produccion.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("data/features_produccion.pkl", "wb") as f:
    pickle.dump(features, f)
with open("data/umbral_produccion.pkl", "wb") as f:
    pickle.dump(mejor_umbral, f)
with open("data/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Plantilla Excel con ejemplos entendibles
plantilla = pd.DataFrame([
    {
        "monto": 150.50, "hora": 14, "tipo_transaccion": "compra",
        "canal": "app_movil", "pais_origen": "CO", "pais_destino": "CO",
        "tarjeta_tipo": "visa", "cliente_edad": 35,
        "cliente_antiguedad_dias": 730, "transacciones_ultimas_24h": 2,
        "monto_promedio_historico": 120.00, "distancia_ultima_transaccion_km": 2.5,
        "es_horario_inusual": 0, "intentos_fallidos_previos": 0
    },
    {
        "monto": 2500.00, "hora": 3, "tipo_transaccion": "transferencia",
        "canal": "web", "pais_origen": "CO", "pais_destino": "RU",
        "tarjeta_tipo": "mastercard", "cliente_edad": 28,
        "cliente_antiguedad_dias": 15, "transacciones_ultimas_24h": 12,
        "monto_promedio_historico": 80.00, "distancia_ultima_transaccion_km": 850.0,
        "es_horario_inusual": 1, "intentos_fallidos_previos": 3
    },
    {
        "monto": 45.00, "hora": 10, "tipo_transaccion": "compra",
        "canal": "cajero", "pais_origen": "MX", "pais_destino": "MX",
        "tarjeta_tipo": "visa", "cliente_edad": 52,
        "cliente_antiguedad_dias": 1825, "transacciones_ultimas_24h": 1,
        "monto_promedio_historico": 95.00, "distancia_ultima_transaccion_km": 0.5,
        "es_horario_inusual": 0, "intentos_fallidos_previos": 0
    }
])

plantilla.to_excel("data/plantilla_fraudlytics.xlsx", index=False, sheet_name="Transacciones")
plantilla.to_csv("data/plantilla_fraudlytics.csv", index=False)

print("✅ Plantilla Excel con 14 columnas bancarias guardada")
print("✅ Modelo guardado")
print("🎉 Listo para producción!")