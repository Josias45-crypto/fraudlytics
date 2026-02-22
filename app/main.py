# ============================================================
# FRAUDLYTICS - Interfaz Web Rediseñada
# Tema: Seguridad Financiera - Dark + Verde Esmeralda
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
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, precision_recall_curve
)

st.set_page_config(
    page_title="Fraudlytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS DISEÑO COMPLETO
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0f0d;
    --bg-secondary: #111916;
    --bg-card: #162118;
    --bg-card-hover: #1c2b1f;
    --accent-green: #00e676;
    --accent-green-dim: #00c853;
    --accent-red: #ff1744;
    --accent-red-dim: #d50000;
    --accent-gold: #ffd600;
    --text-primary: #e8f5e9;
    --text-secondary: #81c784;
    --text-muted: #4caf50;
    --border: #1b5e20;
    --border-bright: #2e7d32;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background: linear-gradient(135deg, #0a0f0d 0%, #0d1f14 50%, #0a0f0d 100%);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── INPUTS ── */
input, textarea, select {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
}

input:focus {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 230, 118, 0.2) !important;
}

/* ── BOTONES ── */
.stButton > button {
    background: linear-gradient(135deg, #00c853, #00e676) !important;
    color: #0a0f0d !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00e676, #69f0ae) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0, 230, 118, 0.3) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 20px !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-green) !important;
    border-bottom: 2px solid var(--accent-green) !important;
}

/* ── MÉTRICAS ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    transition: all 0.2s ease !important;
}

[data-testid="metric-container"]:hover {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 20px rgba(0, 230, 118, 0.1) !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent-green) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── ALERTAS ── */
.stSuccess {
    background: rgba(0, 230, 118, 0.1) !important;
    border: 1px solid var(--accent-green) !important;
    border-radius: 8px !important;
    color: var(--accent-green) !important;
}

.stError {
    background: rgba(255, 23, 68, 0.1) !important;
    border: 1px solid var(--accent-red) !important;
    border-radius: 8px !important;
}

.stWarning {
    background: rgba(255, 214, 0, 0.1) !important;
    border: 1px solid var(--accent-gold) !important;
    border-radius: 8px !important;
}

.stInfo {
    background: rgba(0, 200, 83, 0.08) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
}

/* ── SELECTBOX Y RADIO ── */
[data-testid="stSelectbox"] > div,
[data-testid="stRadio"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border-bright) !important;
    border-radius: 8px !important;
}

/* ── DIVIDER ── */
hr {
    border-color: var(--border) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-green-dim); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# COMPONENTES HTML PERSONALIZADOS
# ============================================================

def hero_login():
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px 0;">
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00e676, #69f0ae);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
            margin-bottom: 8px;
        ">🔍 FRAUDLYTICS</div>
        <div style="
            font-family: 'DM Sans', sans-serif;
            color: #81c784;
            font-size: 0.95rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 32px;
        ">Sistema de detección de fraude financiero</div>
        <div style="
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-bottom: 32px;
        ">
            <div style="text-align:center;">
                <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:1.4rem;">99%</div>
                <div style="color:#4caf50; font-size:0.75rem;">ROC-AUC</div>
            </div>
            <div style="width:1px; background:#1b5e20;"></div>
            <div style="text-align:center;">
                <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:1.4rem;">200K</div>
                <div style="color:#4caf50; font-size:0.75rem;">Transacciones</div>
            </div>
            <div style="width:1px; background:#1b5e20;"></div>
            <div style="text-align:center;">
                <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:1.4rem;">99%</div>
                <div style="color:#4caf50; font-size:0.75rem;">Precisión</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_header(titulo, subtitulo=""):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #162118, #1c2b1f);
        border: 1px solid #1b5e20;
        border-left: 3px solid #00e676;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
    ">
        <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:1.1rem; font-weight:700;">{titulo}</div>
        <div style="color:#81c784; font-size:0.85rem; margin-top:4px;">{subtitulo}</div>
    </div>
    """, unsafe_allow_html=True)

def badge_fraude(cantidad):
    st.markdown(f"""
    <div style="
        background: rgba(255,23,68,0.1);
        border: 1px solid #ff1744;
        border-radius: 8px;
        padding: 16px 24px;
        margin: 16px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <span style="font-size:1.5rem;">🚨</span>
        <div>
            <div style="font-family:'Space Mono',monospace; color:#ff1744; font-size:1.3rem; font-weight:700;">{cantidad} transacciones sospechosas</div>
            <div style="color:#ef9a9a; font-size:0.8rem;">detectadas por el modelo de IA</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MANEJO DE USUARIOS
# ============================================================

USERS_FILE = "app/users.json"

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

for key, val in [("autenticado", False), ("usuario", None), ("nombre", None), ("modo", "login")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# LOGIN / REGISTRO
# ============================================================

if not st.session_state.autenticado:
    col_izq, col_centro, col_der = st.columns([1, 1.5, 1])
    with col_centro:
        hero_login()

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
            st.markdown("<div style='font-family:Space Mono,monospace; color:#00e676; font-size:0.9rem; margin-bottom:16px;'>ACCESO AL SISTEMA</div>", unsafe_allow_html=True)
            usuario = st.text_input("Usuario", placeholder="Ingresa tu usuario")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            if st.button("Entrar →", use_container_width=True, type="primary"):
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
            st.markdown("<div style='color:#4caf50; font-size:0.78rem; text-align:center; margin-top:12px;'>Demo: admin / admin123</div>", unsafe_allow_html=True)

        elif st.session_state.modo == "registro":
            st.markdown("<div style='font-family:Space Mono,monospace; color:#00e676; font-size:0.9rem; margin-bottom:16px;'>CREAR CUENTA</div>", unsafe_allow_html=True)
            nombre = st.text_input("Nombre completo", placeholder="Tu nombre")
            email = st.text_input("Correo electrónico", placeholder="correo@ejemplo.com")
            usuario = st.text_input("Usuario", placeholder="nombre_usuario")
            password = st.text_input("Contraseña", type="password", placeholder="Mínimo 6 caracteres")
            confirmar = st.text_input("Confirmar contraseña", type="password", placeholder="Repite la contraseña")

            if st.button("Crear cuenta →", use_container_width=True, type="primary"):
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
                    usuarios[usuario] = {"nombre": nombre, "email": email, "password": hashed}
                    guardar_usuarios(usuarios)
                    st.success(f"✅ Cuenta creada. Bienvenido, {nombre}!")
                    st.session_state.modo = "login"
                    st.rerun()

# ============================================================
# APP PRINCIPAL
# ============================================================

else:
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #162118, #1c2b1f);
            border: 1px solid #1b5e20;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 20px;
            text-align: center;
        ">
            <div style="font-size:2rem;">👤</div>
            <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:0.9rem;">{st.session_state.nombre}</div>
            <div style="color:#4caf50; font-size:0.75rem;">{st.session_state.usuario}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⬅️ Cerrar sesión", use_container_width=True):
            st.session_state.autenticado = False
            st.session_state.usuario = None
            st.session_state.nombre = None
            st.rerun()

        st.divider()
        st.markdown("<div style='font-family:Space Mono,monospace; color:#00e676; font-size:0.8rem;'>⚙️ CONFIGURACIÓN</div>", unsafe_allow_html=True)
        st.markdown("")

        if os.path.exists("data/plantilla_fraudlytics.xlsx"):
            with open("data/plantilla_fraudlytics.xlsx", "rb") as f:
                st.download_button(
                    label="⬇️ Descargar plantilla Excel",
                    data=f,
                    file_name="plantilla_fraudlytics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        st.divider()
        archivo = st.file_uploader("📂 Subir archivo", type=["csv", "xlsx"])

    # Header principal
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
        padding-bottom: 16px;
        border-bottom: 1px solid #1b5e20;
    ">
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00e676, #69f0ae);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">🔍 FRAUDLYTICS</div>
        <div style="
            background: rgba(0,230,118,0.1);
            border: 1px solid #00e676;
            border-radius: 20px;
            padding: 4px 12px;
            color: #00e676;
            font-size: 0.75rem;
            font-family: 'Space Mono', monospace;
        ">v1.0 PRODUCCIÓN</div>
    </div>
    <div style="color:#81c784; font-size:0.9rem; margin-bottom:24px;">
        Sistema inteligente de detección de fraude en transacciones financieras
    </div>
    """, unsafe_allow_html=True)

    if archivo is not None:
        if archivo.name.endswith(".xlsx"):
            df = pd.read_excel(archivo)
        else:
            df = pd.read_csv(archivo)

        columnas_modelo = [
            "monto", "hora", "tipo_transaccion", "canal",
            "pais_origen", "pais_destino", "tarjeta_tipo",
            "cliente_edad", "cliente_antiguedad_dias",
            "transacciones_ultimas_24h", "monto_promedio_historico",
            "distancia_ultima_transaccion_km", "es_horario_inusual",
            "intentos_fallidos_previos"
        ]

        columnas_faltantes = [c for c in columnas_modelo if c not in df.columns]
        if columnas_faltantes:
            st.error(f"❌ Faltan columnas: {columnas_faltantes}")
            st.info("👈 Descarga la plantilla Excel desde el panel izquierdo.")
            st.stop()

        st.success(f"✅ Archivo cargado correctamente: **{df.shape[0]:,}** transacciones")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Resumen", "📈 Visualización",
            "🤖 Detección de Fraude", "📋 Reporte"
        ])

        with tab1:
            card_header("RESUMEN DEL DATASET", "Vista general de las transacciones cargadas")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total transacciones", f"{df.shape[0]:,}")
            if "es_fraude" in df.columns:
                fraudes = df[df["es_fraude"]==1].shape[0]
                normales = df[df["es_fraude"]==0].shape[0]
                col2.metric("Transacciones normales", f"{normales:,}")
                col3.metric("Fraudes conocidos", f"{fraudes:,}", delta=f"{fraudes/df.shape[0]:.2%}")
            st.markdown("")
            st.dataframe(df.head(20), use_container_width=True)

        with tab2:
            card_header("VISUALIZACIÓN DE PATRONES", "Análisis visual de comportamiento normal vs fraudulento")
            col1, col2 = st.columns(2)

            plt.rcParams.update({
                "figure.facecolor": "#0a0f0d",
                "axes.facecolor": "#162118",
                "axes.edgecolor": "#1b5e20",
                "axes.labelcolor": "#81c784",
                "xtick.color": "#81c784",
                "ytick.color": "#81c784",
                "text.color": "#e8f5e9",
                "grid.color": "#1b5e20",
                "grid.alpha": 0.5
            })

            with col1:
                st.markdown("<div style='color:#00e676; font-family:Space Mono,monospace; font-size:0.85rem; margin-bottom:8px;'>DISTRIBUCIÓN DE MONTOS</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                if "es_fraude" in df.columns:
                    df[df["es_fraude"]==0]["monto"].hist(bins=40, ax=ax, color="#00e676", alpha=0.7, label="Normal")
                    df[df["es_fraude"]==1]["monto"].hist(bins=40, ax=ax, color="#ff1744", alpha=0.8, label="Fraude")
                    ax.legend(facecolor="#162118", edgecolor="#1b5e20", labelcolor="#e8f5e9")
                else:
                    df["monto"].hist(bins=40, ax=ax, color="#00e676", alpha=0.7)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("<div style='color:#00e676; font-family:Space Mono,monospace; font-size:0.85rem; margin-bottom:8px;'>TRANSACCIONES POR CANAL</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                counts = df["canal"].value_counts()
                bars = ax.bar(counts.index, counts.values, color=["#00e676", "#00c853", "#00a152", "#007a37"])
                ax.grid(True, alpha=0.3, axis="y")
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                            str(val), ha='center', va='bottom', color="#e8f5e9", fontsize=9)
                st.pyplot(fig)
                plt.close()

        with tab3:
            card_header("DETECCIÓN DE FRAUDE", "Análisis con modelo de IA entrenado con 200K transacciones")

            with open("data/modelo_produccion.pkl", "rb") as f:
                modelo = pickle.load(f)
            with open("data/scaler_produccion.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("data/umbral_produccion.pkl", "rb") as f:
                umbral = pickle.load(f)
            with open("data/encoders.pkl", "rb") as f:
                encoders = pickle.load(f)

            st.info(f"🏭 Modelo activo | Umbral: {umbral:.2f} | ROC-AUC: 99%")

            df_pred = df[columnas_modelo].copy()
            categoricas = ["tipo_transaccion", "canal", "pais_origen", "pais_destino", "tarjeta_tipo"]
            for col in categoricas:
                if col in encoders:
                    df_pred[col] = df_pred[col].astype(str).map(
                        lambda x, c=col: encoders[c].transform([x])[0]
                        if x in encoders[c].classes_ else 0
                    )

            X = df_pred.values
            X_scaled = scaler.transform(X)
            df["probabilidad_fraude"] = modelo.predict_proba(X_scaled)[:, 1]
            df["prediccion"] = (df["probabilidad_fraude"] >= umbral).astype(int)

            if "es_fraude" in df.columns:
                y = df["es_fraude"].values
                f1 = f1_score(y, df["prediccion"], zero_division=0)
                precision = precision_score(y, df["prediccion"], zero_division=0)
                recall = recall_score(y, df["prediccion"], zero_division=0)
                roc = roc_auc_score(y, df["probabilidad_fraude"])
                accuracy = (df["prediccion"] == y).mean()

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{accuracy:.1%}")
                col2.metric("Precisión", f"{precision:.1%}")
                col3.metric("Recall", f"{recall:.1%}")
                col4.metric("F1-Score", f"{f1:.1%}")
                col5.metric("ROC-AUC", f"{roc:.1%}")

                st.markdown("")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div style='color:#00e676; font-family:Space Mono,monospace; font-size:0.85rem; margin-bottom:8px;'>MATRIZ DE CONFUSIÓN</div>", unsafe_allow_html=True)
                    cm = confusion_matrix(y, df["prediccion"])
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d",
                                cmap=sns.light_palette("#00e676", as_cmap=True),
                                xticklabels=["Normal", "Fraude"],
                                yticklabels=["Normal", "Fraude"],
                                ax=ax, linewidths=1, linecolor="#0a0f0d")
                    ax.set_facecolor("#162118")
                    fig.patch.set_facecolor("#0a0f0d")
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.markdown("<div style='color:#00e676; font-family:Space Mono,monospace; font-size:0.85rem; margin-bottom:8px;'>CURVA PRECISION-RECALL</div>", unsafe_allow_html=True)
                    p_curve, r_curve, _ = precision_recall_curve(y, df["probabilidad_fraude"])
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(r_curve, p_curve, color="#00e676", lw=2)
                    ax.fill_between(r_curve, p_curve, alpha=0.15, color="#00e676")
                    ax.set_facecolor("#162118")
                    fig.patch.set_facecolor("#0a0f0d")
                    ax.grid(True, alpha=0.3, color="#1b5e20")
                    st.pyplot(fig)
                    plt.close()

            fraudes_detectados = df[df["prediccion"] == 1].shape[0]
            badge_fraude(fraudes_detectados)

            st.dataframe(
                df[df["prediccion"] == 1][columnas_modelo + ["probabilidad_fraude"]].sort_values(
                    "probabilidad_fraude", ascending=False
                ).style.background_gradient(subset=["probabilidad_fraude"], cmap="Reds"),
                use_container_width=True
            )

        with tab4:
            card_header("REPORTE DE FRAUDES", "Exporta las transacciones sospechosas detectadas")
            if "prediccion" in df.columns:
                fraudes_df = df[df["prediccion"] == 1]
                st.metric("Fraudes detectados para exportar", fraudes_df.shape[0])
                csv = fraudes_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Descargar reporte CSV",
                    data=csv,
                    file_name="fraudes_detectados.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("Primero ve a la pestaña Detección de Fraude.")

    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="background:#162118; border:1px solid #1b5e20; border-radius:10px; padding:20px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:8px;">📥</div>
                <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:0.85rem;">PASO 1</div>
                <div style="color:#81c784; font-size:0.8rem; margin-top:8px;">Descarga la plantilla Excel desde el panel izquierdo</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background:#162118; border:1px solid #1b5e20; border-radius:10px; padding:20px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:8px;">✏️</div>
                <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:0.85rem;">PASO 2</div>
                <div style="color:#81c784; font-size:0.8rem; margin-top:8px;">Llena el archivo con tus datos de transacciones</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="background:#162118; border:1px solid #1b5e20; border-radius:10px; padding:20px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:8px;">🤖</div>
                <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:0.85rem;">PASO 3</div>
                <div style="color:#81c784; font-size:0.8rem; margin-top:8px;">Sube el archivo y el modelo detectará fraudes</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#162118; border:1px solid #1b5e20; border-radius:10px; padding:24px;">
            <div style="font-family:'Space Mono',monospace; color:#00e676; font-size:0.9rem; margin-bottom:16px;">📋 FORMATO REQUERIDO</div>
            <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
                <tr style="border-bottom:1px solid #1b5e20;">
                    <th style="color:#00e676; padding:8px; text-align:left;">Columna</th>
                    <th style="color:#00e676; padding:8px; text-align:left;">Descripción</th>
                    <th style="color:#00e676; padding:8px; text-align:left;">Ejemplo</th>
                </tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">monto</td><td style="color:#4caf50; padding:8px;">Monto de la transacción</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">150.50</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">hora</td><td style="color:#4caf50; padding:8px;">Hora del día (0-23)</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">14</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">tipo_transaccion</td><td style="color:#4caf50; padding:8px;">compra, retiro, transferencia, pago_servicio</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">compra</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">canal</td><td style="color:#4caf50; padding:8px;">web, app_movil, cajero, sucursal</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">app_movil</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">pais_origen</td><td style="color:#4caf50; padding:8px;">País donde se hizo la transacción</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">CO</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">pais_destino</td><td style="color:#4caf50; padding:8px;">País destino del dinero</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">US</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">tarjeta_tipo</td><td style="color:#4caf50; padding:8px;">visa, mastercard, amex, diners</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">visa</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">cliente_edad</td><td style="color:#4caf50; padding:8px;">Edad del cliente</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">35</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">cliente_antiguedad_dias</td><td style="color:#4caf50; padding:8px;">Días como cliente del banco</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">730</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">transacciones_ultimas_24h</td><td style="color:#4caf50; padding:8px;">Transacciones realizadas hoy</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">2</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">monto_promedio_historico</td><td style="color:#4caf50; padding:8px;">Gasto promedio histórico</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">120.00</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">distancia_ultima_transaccion_km</td><td style="color:#4caf50; padding:8px;">Distancia desde última compra</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">2.5</td></tr>
                <tr style="border-bottom:1px solid #1b5e20;"><td style="color:#81c784; padding:8px;">es_horario_inusual</td><td style="color:#4caf50; padding:8px;">1 si es de madrugada, 0 si no</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">0</td></tr>
                <tr><td style="color:#81c784; padding:8px;">intentos_fallidos_previos</td><td style="color:#4caf50; padding:8px;">Intentos fallidos antes</td><td style="color:#e8f5e9; padding:8px; font-family:monospace;">0</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)