# ============================================================
# FRAUDLYTICS - Interfaz Web con Login y Registro simple
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import bcrypt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Fraudlytics",
    page_icon="🔍",
    layout="wide"
)

USERS_FILE = "app/users.json"

# ============================================================
# MANEJO DE USUARIOS
# ============================================================

def cargar_usuarios():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {
        "admin": {
            "nombre": "Administrador",
            "email": "admin@fraudlytics.com",
            "password": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        }
    }

def guardar_usuarios(usuarios):
    with open(USERS_FILE, "w") as f:
        json.dump(usuarios, f)

def verificar_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# ============================================================
# SESSION STATE
# ============================================================

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
if "usuario" not in st.session_state:
    st.session_state.usuario = None
if "nombre" not in st.session_state:
    st.session_state.nombre = None
if "modo" not in st.session_state:
    st.session_state.modo = "login"

# ============================================================
# PANTALLA DE LOGIN / REGISTRO
# ============================================================

if not st.session_state.autenticado:

    col_izq, col_centro, col_der = st.columns([1, 2, 1])

    with col_centro:
        st.markdown("## 🔍 Fraudlytics")
        st.markdown("##### Sistema de detección de fraude financiero")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Iniciar sesión", use_container_width=True):
                st.session_state.modo = "login"
        with col2:
            if st.button("📝 Registrarse", use_container_width=True):
                st.session_state.modo = "registro"

        st.divider()

        usuarios = cargar_usuarios()

        # LOGIN
        if st.session_state.modo == "login":
            st.subheader("Iniciar sesión")
            usuario = st.text_input("🆔 Usuario")
            password = st.text_input("🔒 Contraseña", type="password")

            if st.button("Entrar", use_container_width=True, type="primary"):
                if usuario in usuarios:
                    if verificar_password(password, usuarios[usuario]["password"]):
                        st.session_state.autenticado = True
                        st.session_state.usuario = usuario
                        st.session_state.nombre = usuarios[usuario]["nombre"]
                        st.rerun()
                    else:
                        st.error("❌ Contraseña incorrecta")
                else:
                    st.error("❌ Usuario no encontrado")

            st.caption("Usuario de prueba: **admin** | Contraseña: **admin123**")

        # REGISTRO
        elif st.session_state.modo == "registro":
            st.subheader("Crear cuenta nueva")
            nombre = st.text_input("👤 Nombre completo")
            email = st.text_input("📧 Correo electrónico")
            usuario = st.text_input("🆔 Nombre de usuario")
            password = st.text_input("🔒 Contraseña", type="password")
            confirmar = st.text_input("🔒 Confirmar contraseña", type="password")

            if st.button("Crear cuenta", use_container_width=True, type="primary"):
                if not nombre or not email or not usuario or not password:
                    st.error("❌ Todos los campos son obligatorios")
                elif "@" not in email:
                    st.error("❌ El correo no es válido")
                elif len(password) < 6:
                    st.error("❌ La contraseña debe tener al menos 6 caracteres")
                elif password != confirmar:
                    st.error("❌ Las contraseñas no coinciden")
                elif usuario in usuarios:
                    st.error("❌ El usuario ya existe")
                else:
                    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                    usuarios[usuario] = {
                        "nombre": nombre,
                        "email": email,
                        "password": hashed
                    }
                    guardar_usuarios(usuarios)
                    st.success(f"✅ Cuenta creada. Ya puedes iniciar sesión, {nombre}!")
                    st.session_state.modo = "login"
                    st.rerun()

# ============================================================
# APP PRINCIPAL
# ============================================================

else:
    st.sidebar.title(f"👤 {st.session_state.nombre}")
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.autenticado = False
        st.session_state.usuario = None
        st.session_state.nombre = None
        st.rerun()

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
                    recall = recall_score(y, df["prediccion"])
                    roc = roc_auc_score(y, df["probabilidad_fraude"])
                    accuracy = (df["prediccion"] == y).mean()

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Accuracy", f"{accuracy:.2%}")
                    col2.metric("Precisión", f"{precision:.2%}")
                    col3.metric("Recall", f"{recall:.2%}")
                    col4.metric("F1-Score", f"{f1:.2%}")
                    col5.metric("ROC-AUC", f"{roc:.2%}")

                    if recall < 0.7:
                        st.warning("⚠️ El modelo está dejando pasar muchos fraudes.")
                    elif precision < 0.7:
                        st.warning("⚠️ El modelo está marcando demasiadas transacciones normales como fraude.")
                    else:
                        st.success("✅ El modelo tiene un buen balance.")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Matriz de Confusión")
                        cm = confusion_matrix(y, df["prediccion"])
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                    xticklabels=["Normal", "Fraude"],
                                    yticklabels=["Normal", "Fraude"], ax=ax)
                        st.pyplot(fig)
                    with col2:
                        st.subheader("Curva Precision-Recall")
                        p_curve, r_curve, _ = precision_recall_curve(y, df["probabilidad_fraude"])
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.plot(r_curve, p_curve, color="blue", lw=2)
                        ax.fill_between(r_curve, p_curve, alpha=0.2, color="blue")
                        st.pyplot(fig)

                    st.subheader("Transacciones sospechosas")
                    st.dataframe(df[df["prediccion"] == 1][["Amount", "probabilidad_fraude", "Class"]].sort_values(
                        "probabilidad_fraude", ascending=False))

        with tab4:
            st.header("📋 Reporte")
            if "prediccion" in df.columns:
                csv = df[df["prediccion"] == 1].to_csv(index=False)
                st.download_button(
                    "⬇️ Descargar CSV de fraudes detectados",
                    data=csv,
                    file_name="fraudes_detectados.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Primero ve a la pestaña Detección de Fraude.")

    else:
        st.info("👈 Sube un archivo CSV para comenzar.")