# ============================================================
# FRAUDLYTICS - Interfaz Web con Autenticación
# Streamlit + Streamlit Authenticator
# ============================================================

import streamlit as st
import streamlit_authenticator as stauth
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

# ============================================================
# AUTENTICACIÓN
# ============================================================

credentials = {
    "usernames": {
        "admin": {
            "name": "Administrador",
            "password": stauth.Hasher(["admin123"]).generate()[0]
        },
        "analista": {
            "name": "Analista",
            "password": stauth.Hasher(["analista123"]).generate()[0]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "fraudlytics_cookie",
    "clave_secreta_fraudlytics",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Login - Fraudlytics", "main")

if authentication_status == False:
    st.error("❌ Usuario o contraseña incorrectos")
    st.stop()

elif authentication_status == None:
    st.warning("👆 Ingresa tu usuario y contraseña para continuar")
    st.info("**Usuario:** admin | **Contraseña:** admin123")
    st.stop()

elif authentication_status:

    # ============================================================
    # APP PRINCIPAL
    # ============================================================

    st.sidebar.title(f"👤 Bienvenido, {name}")
    authenticator.logout("Cerrar sesión", "sidebar")

    st.title("🔍 Fraudlytics")
    st.subheader("Sistema inteligente de detección de fraude en transacciones financieras")

    st.sidebar.title("⚙️ Configuración")
    archivo = st.sidebar.file_uploader("📂 Subir CSV de transacciones", type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo)
        st.success(f"✅ Archivo cargado: {df.shape[0]} transacciones")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Resumen",
            "📈 Visualización",
            "🤖 Detección de Fraude",
            "📋 Reporte"
        ])

        with tab1:
            st.header("📊 Resumen del dataset")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total transacciones", f"{df.shape[0]:,}")
            if "Class" in df.columns:
                fraudes = df[df["Class"] == 1].shape[0]
                normales = df[df["Class"] == 0].shape[0]
                col2.metric("Transacciones normales", f"{normales:,}")
                col3.metric("Fraudes", f"{fraudes:,}", delta=f"{fraudes/df.shape[0]:.2%}")
            st.dataframe(df.head(20))
            st.dataframe(df.describe())

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
                    st.pyplot(fig)
                with col2:
                    st.subheader("Conteo de transacciones")
                    fig, ax = plt.subplots()
                    counts = df["Class"].value_counts()
                    ax.bar(["Normal", "Fraude"], counts.values, color=["steelblue", "red"])
                    st.pyplot(fig)

        with tab3:
            st.header("🤖 Detección de Fraude")
            if "Class" in df.columns:
                features = [col for col in df.columns if col.startswith("V")]
                if len(features) > 0:
                    from sklearn.metrics import (
                        f1_score, precision_score, recall_score,
                        confusion_matrix, roc_auc_score, precision_recall_curve
                    )
                    from xgboost import XGBClassifier

                    algoritmo = st.selectbox(
                        "🤖 Elige el algoritmo:",
                        ["Regresión Logística", "XGBoost"]
                    )

                    X = df[features].values
                    y = df["Class"].values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    if algoritmo == "Regresión Logística":
                        modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
                    else:
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

                    st.success("✅ Modelo entrenado")

                    f1 = f1_score(y, df["prediccion"])
                    precision = precision_score(y, df["prediccion"])