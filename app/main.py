# ============================================================
# FRAUDLYTICS - Interfaz Web
# Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================================================
# CONFIGURACIÓN DE LA APP
# ============================================================

st.set_page_config(
    page_title="Fraudlytics",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fraudlytics")
st.subheader("Sistema inteligente de detección de fraude en transacciones financieras")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("Subír.")

archivo = st.sidebar.file_uploader("📂 Subir CSV de transacciones", type=["csv"])

# ============================================================
# CARGA DE DATOS
# ============================================================

if archivo is not None:
    df = pd.read_csv(archivo)
    st.success(f"✅ Archivo cargado: {df.shape[0]} transacciones y {df.shape[1]} columnas")

    # ============================================================
    # TABS DE LA APP
    # ============================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resumen", 
        "📈 Visualización", 
        "🤖 Detección de Fraude",
        "📋 Reporte"
    ])

    # ============================================================
    # TAB 1 - RESUMEN
    # ============================================================

    with tab1:
        st.header("📊 Resumen del dataset")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total transacciones", f"{df.shape[0]:,}")

        if "Class" in df.columns:
            fraudes = df[df["Class"] == 1].shape[0]
            normales = df[df["Class"] == 0].shape[0]
            col2.metric("Transacciones normales", f"{normales:,}")
            col3.metric("Transacciones fraudulentas", f"{fraudes:,}", delta=f"{fraudes/df.shape[0]:.2%}")

        st.subheader("Vista previa de los datos")
        st.dataframe(df.head(20))

        st.subheader("Estadísticas generales")
        st.dataframe(df.describe())

    # ============================================================
    # TAB 2 - VISUALIZACIÓN
    # ============================================================

    with tab2:
        st.header("📈 Visualización de patrones")

        if "Class" in df.columns and "Amount" in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribución de montos")
                fig, ax = plt.subplots()
                df[df["Class"] == 0]["Amount"].hist(bins=50, ax=ax, color="steelblue", alpha=0.7, label="Normal")
                df[df["Class"] == 1]["Amount"].hist(bins=50, ax=ax, color="red", alpha=0.7, label="Fraude")
                ax.legend()
                ax.set_xlabel("Monto")
                ax.set_ylabel("Frecuencia")
                st.pyplot(fig)

            with col2:
                st.subheader("Conteo de transacciones")
                fig, ax = plt.subplots()
                counts = df["Class"].value_counts()
                ax.bar(["Normal", "Fraude"], counts.values, color=["steelblue", "red"])
                ax.set_ylabel("Cantidad")
                st.pyplot(fig)

            st.subheader("Mapa de calor de correlaciones")
            fig, ax = plt.subplots(figsize=(12, 6))
            features = [col for col in df.columns if col.startswith("V")][:10]
            sns.heatmap(df[features + ["Amount", "Class"]].corr(), annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # ============================================================
    # TAB 3 - DETECCIÓN DE FRAUDE
    # ============================================================

    with tab3:
        st.header("🤖 Detección de Fraude")

        if "Class" in df.columns:
            features = [col for col in df.columns if col.startswith("V")]

            if len(features) > 0:
                st.info("Entrenando modelo de detección en tiempo real...")

                X = df[features].values
                y = df["Class"].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Selector de modelo
                algoritmo = st.selectbox(
                    "🤖 Elige el algoritmo de detección:",
                    ["Regresión Logística", "XGBoost"]
                )

                if algoritmo == "Regresión Logística":
                    modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
                    modelo.fit(X_scaled, y)
                else:
                    from xgboost import XGBClassifier
                    modelo = XGBClassifier(
                        scale_pos_weight=len(y[y==0])/len(y[y==1]),
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric="logloss"
                    )
                    modelo.fit(X_scaled, y)

                df["probabilidad_fraude"] = modelo.predict_proba(X_scaled)[:, 1]
                df["prediccion"] = modelo.predict(X_scaled)

                st.success("✅ Modelo entrenado correctamente")

                # Métricas principales
                from sklearn.metrics import (
                    f1_score, precision_score, recall_score,
                    confusion_matrix, roc_auc_score
                )

                f1 = f1_score(y, df["prediccion"])
                precision = precision_score(y, df["prediccion"])
                recall = recall_score(y, df["prediccion"])
                roc = roc_auc_score(y, df["probabilidad_fraude"])
                accuracy = (df["prediccion"] == y).mean()

                st.subheader("📊 Métricas del modelo")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Precisión", f"{precision:.2%}", 
                           help="De los fraudes detectados, ¿cuántos eran reales?")
                col3.metric("Recall", f"{recall:.2%}",
                           help="De todos los fraudes reales, ¿cuántos detectó?")
                col4.metric("F1-Score", f"{f1:.2%}",
                           help="Equilibrio entre Precisión y Recall")
                col5.metric("ROC-AUC", f"{roc:.2%}",
                           help="Qué tan bien separa fraudes de normales")

                # Interpretación automática
                st.subheader("🧠 Interpretación")
                if recall < 0.7:
                    st.warning("⚠️ El modelo está dejando pasar muchos fraudes. Considera reentrenar con más datos.")
                elif precision < 0.7:
                    st.warning("⚠️ El modelo está marcando demasiadas transacciones normales como fraude.")
                else:
                    st.success("✅ El modelo tiene un buen balance entre precisión y recall.")

                # Matriz de confusión
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Matriz de Confusión")
                    cm = confusion_matrix(y, df["prediccion"])
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["Normal", "Fraude"],
                                yticklabels=["Normal", "Fraude"], ax=ax)
                    ax.set_ylabel("Real")
                    ax.set_xlabel("Predicho")
                    st.pyplot(fig)

                with col2:
                    st.subheader("Curva Precision-Recall")
                    from sklearn.metrics import precision_recall_curve
                    precision_curve, recall_curve, _ = precision_recall_curve(y, df["probabilidad_fraude"])
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(recall_curve, precision_curve, color="blue", lw=2)
                    ax.fill_between(recall_curve, precision_curve, alpha=0.2, color="blue")
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    ax.set_title("Curva Precision-Recall")
                    st.pyplot(fig)

                st.subheader("Transacciones sospechosas detectadas")
                sospechosas = df[df["prediccion"] == 1][["Amount", "probabilidad_fraude", "Class"]].sort_values(
                    "probabilidad_fraude", ascending=False
                )
                st.dataframe(sospechosas)

    # ============================================================
    # TAB 4 - REPORTE
    # ============================================================

    with tab4:
        st.header("📋 Reporte de resultados")

        if "prediccion" in df.columns:
            st.subheader("Descargar reporte de transacciones sospechosas")
            sospechosas_csv = df[df["prediccion"] == 1].to_csv(index=False)
            st.download_button(
                label="⬇️ Descargar CSV de fraudes detectados",
                data=sospechosas_csv,
                file_name="fraudes_detectados.csv",
                mime="text/csv"
            )
        else:
            st.warning("Primero ve a la pestaña 'Detección de Fraude' para analizar los datos.")

else:
    st.info("👈 Sube un archivo CSV en el panel izquierdo para comenzar el análisis.")
    st.markdown("""
    ### ¿Cómo usar Fraudlytics?
    1. Sube tu archivo CSV de transacciones desde el panel izquierdo
    2. Explora el resumen y estadísticas de tus datos
    3. Visualiza los patrones de comportamiento
    4. Detecta transacciones fraudulentas automáticamente
    5. Descarga el reporte con los fraudes detectados
    """)