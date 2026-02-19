# ============================================================
# FRAUDLYTICS - PASO 5: Validación del Modelo
# Scikit-learn con métricas especializadas
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import seaborn as sns

# ============================================================
# CARGA DE DATOS
# ============================================================

print("📦 Cargando datos con texto...")
df = pd.read_csv("data/datos_con_texto.csv")
print(f"✅ Datos cargados: {df.shape[0]} filas")

features_numericas = [col for col in df.columns if col.startswith("V")]
features_lsa = [col for col in df.columns if col.startswith("lsa_")]
features = features_numericas + features_lsa

X = df[features].values
y = df["Class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"\n📊 Balance del dataset:")
print(f"   Normales:  {(y == 0).sum()} ({(y == 0).mean():.1%})")
print(f"   Fraudes:   {(y == 1).sum()} ({(y == 1).mean():.1%})")

# ============================================================
# BLOQUE A - VALIDACIÓN CRUZADA: StratifiedKFold
# ============================================================

print("\n🔁 Aplicando StratifiedKFold (5 folds)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo = LogisticRegression(max_iter=1000, class_weight="balanced")

f1_scores = []
roc_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    f1_scores.append(f1)
    roc_scores.append(roc)

    print(f"   Fold {fold+1} - F1: {f1:.4f} | ROC-AUC: {roc:.4f}")

print(f"\n✅ F1 promedio:      {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"✅ ROC-AUC promedio: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")

# ============================================================
# BLOQUE B - MÉTRICAS FINALES
# ============================================================

print("\n📋 Reporte final del modelo...")
modelo.fit(X, y)
y_pred_final = modelo.predict(X)
y_prob_final = modelo.predict_proba(X)[:, 1]

print(classification_report(y, y_pred_final, target_names=["Normal", "Fraude"]))

# ============================================================
# BLOQUE C - GRÁFICAS DE VALIDACIÓN
# ============================================================

print("\n🎨 Generando gráficas de validación...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fraudlytics - Validación del Modelo", fontsize=16)

# Gráfica 1: Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y, y_prob_final)
axes[0].plot(recall, precision, color="blue", lw=2)
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Curva Precision-Recall")
axes[0].fill_between(recall, precision, alpha=0.2, color="blue")

# Gráfica 2: Curva ROC
fpr, tpr, _ = roc_curve(y, y_prob_final)
axes[1].plot(fpr, tpr, color="green", lw=2, label=f"AUC = {np.mean(roc_scores):.4f}")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
axes[1].set_xlabel("Tasa de Falsos Positivos")
axes[1].set_ylabel("Tasa de Verdaderos Positivos")
axes[1].set_title("Curva ROC")
axes[1].legend()

# Gráfica 3: Matriz de confusión
cm = confusion_matrix(y, y_pred_final)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraude"],
            yticklabels=["Normal", "Fraude"], ax=axes[2])
axes[2].set_title("Matriz de Confusión")
axes[2].set_ylabel("Real")
axes[2].set_xlabel("Predicho")

plt.tight_layout()
plt.savefig("data/validacion_modelo.png", dpi=150)
plt.show()
print("✅ Gráficas guardadas")

print("\n🎉 Paso 5 completado. ¡Fraudlytics está listo!")