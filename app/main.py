# ============================================================
# FRAUDLYTICS - Interfaz Web con Login y Registro
# ============================================================

import pickle
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
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, precision_recall_curve
)
from xgboost import XGBClassifier

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

    # Sidebar con plantilla y uploader
    st.sidebar.title("⚙️ Configuración")

    if os.path.exists("data/plantilla_fraudlytics.xlsx"):
        with open("data/plantilla_fraudlytics.xlsx", "rb") as f:
            st.sidebar.download_button(
                label="⬇️ Descargar plantilla Excel",
                data=f,
                file_name="plantilla_fraudlytics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Descarga la plantilla Excel con el formato correcto"
        )
        st.sidebar.divider()

    archivo = st.sidebar.file_uploader("📂 Subir CSV de transacciones", type=["csv"])

    if archivo is not None:

        df = pd.read_csv(archivo)

        # Cargar features del modelo
        with open("data/features_produccion.pkl", "rb") as f:
            features_modelo = pickle.load(f)

        # Validar columnas
        columnas_disponibles = [c for c in features_modelo if c in df.columns]
        columnas_faltantes = [c for c in features_modelo if c not in df.columns]

        if len(columnas_disponibles) < 50:
            st.error("❌ El CSV no tiene el formato correcto para el modelo de producción.")
            st.warning(f"Faltan {len(columnas_faltantes)} columnas.")
            st.info("👈 Descarga la plantilla desde el panel izquierdo para ver el formato correcto.")
            with st.expander("Ver primeras 10 columnas faltantes"):
                st.write(columnas_faltantes[:10])
            st.stop()

        st.success(f"✅ Archivo cargado: {df.shape[0]:,} transacciones")

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
            if "isFraud" in df.columns:
                fraudes = df[df["isFraud"] == 1].shape[0]
                normales = df[df["isFraud"] == 0].shape[0]
                col2.metric("Transacciones normales", f"{normales:,}")
                col3.metric("Fraudes", f"{fraudes:,}", delta=f"{fraudes/df.shape[0]:.2%}")
            elif "Class" in df.columns:
                fraudes = df[df["Class"] == 1].shape[0]
                normales = df[df["Class"] == 0].shape[0]
                col2.metric("Transacciones normales", f"{normales:,}")
                col3.metric("Fraudes", f"{fraudes:,}", delta=f"{fraudes/df.shape[0]:.2%}")
            st.dataframe(df.head(20))
            st.dataframe(df.describe())

        with tab2:
            st.header("📈 Visualización de patrones")
            col_monto = "TransactionAmt" if "TransactionAmt" in df.columns else "Amount" if "Amount" in df.columns else None
            col_target = "isFraud" if "isFraud" in df.columns else "Class" if "Class" in df.columns else None

            if col_monto and col_target:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Distribución de montos")
                    fig, ax = plt.subplots()
                    df[df[col_target] == 0][col_monto].hist(bins=50, ax=ax, color="steelblue", alpha=0.7, label="Normal")
                    df[df[col_target] == 1][col_monto].hist(bins=50, ax=ax, color="red", alpha=0.7, label="Fraude")
                    ax.legend()
                    st.pyplot(fig)
                with col2:
                    st.subheader("Conteo de transacciones")
                    fig, ax = plt.subplots()
                    counts = df[col_target].value_counts()
                    ax.bar(["Normal", "Fraude"], counts.values, color=["steelblue", "red"])
                    st.pyplot(fig)

        with tab3:
            st.header("🤖 Detección de Fraude")

            # Cargar modelo de producción
            with open("data/modelo_produccion.pkl", "rb") as f:
                modelo = pickle.load(f)
            with open("data/scaler_produccion.pkl", "rb") as f:
                scaler_prod = pickle.load(f)
            with open("data/umbral_produccion.pkl", "rb") as f:
                umbral = pickle.load(f)

            st.info(f"🏭 Usando modelo de producción IEEE-CIS | Umbral: {umbral:.2f} | ROC-AUC: 94.9%")

            # Preparar datos con las features del modelo
            X_prod = np.zeros((len(df), len(features_modelo)))
            for i, feat in enumerate(features_modelo):
                if feat in df.columns:
                    X_prod[:, i] = pd.to_numeric(df[feat], errors="coerce").fillna(0).values

            X_prod_scaled = scaler_prod.transform(X_prod)
            df["probabilidad_fraude"] = modelo.predict_proba(X_prod_scaled)[:, 1]
            df["prediccion"] = (df["probabilidad_fraude"] >= umbral).astype(int)

            st.success("✅ Análisis completado")

            # Métricas si hay columna target
            col_target = "isFraud" if "isFraud" in df.columns else "Class" if "Class" in df.columns else None

            if col_target:
                y = df[col_target].values
                f1 = f1_score(y, df["prediccion"])
                precision = precision_score(y, df["prediccion"], zero_division=0)
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

            fraudes_detectados = df[df["prediccion"] == 1].shape[0]
            st.subheader(f"🚨 Transacciones sospechosas detectadas: {fraudes_detectados}")
            cols_mostrar = ["probabilidad_fraude", "prediccion"]
            if "TransactionAmt" in df.columns:
                cols_mostrar = ["TransactionAmt"] + cols_mostrar
            elif "Amount" in df.columns:
                cols_mostrar = ["Amount"] + cols_mostrar
            st.dataframe(df[df["prediccion"] == 1][cols_mostrar].sort_values(
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
        st.markdown("""
        ### ¿Cómo usar Fraudlytics?
        1. **Descarga la plantilla** desde el panel izquierdo para ver el formato correcto
        2. **Prepara tu CSV** con el mismo formato que la plantilla
        3. **Sube tu CSV** y el modelo analizará automáticamente las transacciones
        4. **Revisa los resultados** en las pestañas de detección y reporte
        """)