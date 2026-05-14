import streamlit as st
import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq  # <-- IMPORTANTE: Librería de Groq añadida

# 1. Configuración Inicial
load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 2. Inicializar Cliente de Groq (Llama 3)
# Esta línea faltaba y por eso daba el error de 'client not defined'
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Gym IA - Pro & AI", layout="wide")

# --- FUNCIONES DE APOYO ---
def obtener_ultimo_entreno(grupo_muscular):
    try:
        response = supabase.table("registros_gym").select("*").eq("grupo_muscular", grupo_muscular).order("fecha", desc=True).limit(20).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            ultima_fecha = df['fecha'].max()
            ultimos = df[df['fecha'] == ultima_fecha]
            return [{"nombre": f['ejercicio'], "series": f['datos_series']} for _, f in ultimos.iterrows()]
    except: pass
    return []

def calcular_1rm_func(peso, reps):
    if reps <= 0: return 0
    if reps == 1: return peso
    return peso / (1.0278 - (0.0278 * reps))

def generar_respuesta_llama(prompt):
    try:
        # Llamada al motor de Llama 3 en Groq
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error con Llama: {e}"

# --- ESTRUCTURA DE PESTAÑAS ---
tab1, tab2, tab3, tab4 = st.tabs(["📝 Registrar", "📅 Historial", "📈 Evolución", "🤖 Coach IA"])

# --- PESTAÑA 1: REGISTRO ---
with tab1:
    st.title("🏋️‍♂️ Nuevo Entrenamiento")
    col_fecha, col_grupo = st.columns(2)
    with col_fecha:
        fecha_entreno = st.date_input("Fecha", datetime.now(), key="fecha_registro")
    with col_grupo:
        grupo = st.selectbox("Split", ["Espalda + Bíceps", "Pecho + Hombro + Tríceps", "Pierna", "Hombro + Tríceps + Bíceps", "Espalda + Pecho"], key="split_registro")

    if 'lista_ejercicios' not in st.session_state:
        st.session_state.lista_ejercicios = []

    c1, c2 = st.columns(2)
    if c1.button("🔄 Cargar últimos pesos"):
        st.session_state.lista_ejercicios = obtener_ultimo_entreno(grupo)
    if c2.button("➕ Añadir Ejercicio"):
        st.session_state.lista_ejercicios.append({"nombre": "", "series": [{"peso": 0.0, "repes": 0}]})

    for i, ej in enumerate(st.session_state.lista_ejercicios):
        with st.expander(f"Ejercicio: {ej['nombre']}", expanded=True):
            ej['nombre'] = st.text_input(f"Nombre", value=ej['nombre'], key=f"n_{i}")
            s_col1, s_col2 = st.columns(2)
            if s_col1.button(f"➕ Serie", key=f"as_{i}"):
                ej['series'].append({"peso": ej['series'][-1]['peso'], "repes": ej['series'][-1]['repes']})
            if s_col2.button(f"➖ Serie", key=f"rs_{i}") and len(ej['series']) > 1:
                ej['series'].pop()
            
            for j, serie in enumerate(ej['series']):
                cx, cy, cz, cw = st.columns([0.5, 1.5, 1.5, 1.5])
                cx.markdown(f"<br>**S{j+1}**", unsafe_allow_html=True)
                serie['peso'] = cy.number_input("kg", key=f"p_{i}_{j}", value=float(serie['peso']), step=0.5)
                serie['repes'] = cz.number_input("reps", key=f"r_{i}_{j}", value=int(serie['repes']), step=1)
                est_1rm = calcular_1rm_func(serie['peso'], serie['repes'])
                cw.metric("Est. 1RM", f"{round(est_1rm, 1)} kg")

    if st.button("💾 GUARDAR SESIÓN", type="primary", use_container_width=True):
        datos = [{"fecha": str(fecha_entreno), "grupo_muscular": grupo, "ejercicio": e['nombre'], "datos_series": e['series']} for e in st.session_state.lista_ejercicios]
        supabase.table("registros_gym").insert(datos).execute()
        st.success("¡Guardado!")
        st.balloons()

# --- PESTAÑA 2: HISTORIAL ---
with tab2:
    st.title("📅 Consultar Pasado")
    fecha_busqueda = st.date_input("Selecciona una fecha:", datetime.now())
    res_historial = supabase.table("registros_gym").select("*").eq("fecha", str(fecha_busqueda)).execute()
    if res_historial.data:
        df_hist = pd.DataFrame(res_historial.data)
        st.subheader(f"Entrenamiento: {df_hist['grupo_muscular'].iloc[0]}")
        for _, fila in df_hist.iterrows():
            st.markdown(f"### 🔘 {fila['ejercicio'].upper()}")
            cols = st.columns(len(fila['datos_series']))
            for idx, s in enumerate(fila['datos_series']):
                cols[idx].code(f"S{idx+1}: {s['peso']}kg x {s['repes']}")
            st.divider()

# --- PESTAÑA 3: EVOLUCIÓN ---
with tab3:
    st.title("📈 Evolución Real")
    res_graf = supabase.table("registros_gym").select("*").order("fecha").execute()
    if res_graf.data:
        df_todo = pd.DataFrame(res_graf.data)
        sel_ej = st.selectbox("Elige ejercicio:", df_todo['ejercicio'].unique(), key="sel_grafica")
        df_ejercicio = df_todo[df_todo['ejercicio'] == sel_ej].copy()
        df_ejercicio['fecha'] = pd.to_datetime(df_ejercicio['fecha'])
        
        def datos_progresion(lista_series):
            if not lista_series: return 0, 0
            mejor_serie = max(lista_series, key=lambda x: float(x['peso']))
            return float(mejor_serie['peso']), int(mejor_serie['repes'])
        
        df_ejercicio[['peso_max', 'repes_max']] = df_ejercicio['datos_series'].apply(lambda x: pd.Series(datos_progresion(x)))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df_ejercicio['fecha'], y=df_ejercicio['peso_max'], name="Peso (kg)", marker_color='rgba(255, 75, 75, 0.6)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_ejercicio['fecha'], y=df_ejercicio['repes_max'], name="Repeticiones", mode='lines+markers', line=dict(color='#00ff00', width=3)), secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# --- PESTAÑA 4: COACH IA (LLAMA 3) ---
with tab4:
    st.title("🤖 Coach IA con Llama 3")
    st.write("Analizando tus progresos con la potencia de Llama 3 (Groq).")

    if st.button("🔍 Analizar mi progresión con Llama"):
        res = supabase.table("registros_gym").select("*").order("fecha", desc=True).limit(50).execute()
        
        if res.data:
            with st.spinner("Llama 3 está analizando tus series..."):
                contexto_entrenos = str(res.data)
                prompt = f"""
                Analiza estos entrenos: {contexto_entrenos}
                Dame un informe ultra-conciso (máximo 150 palabras).
                Usa puntos clave (bullet points).
                Dime: 
                1. Ejercicio estancado.
                2. Recomendación de peso para la próxima vez.
                3. Un consejo técnico rápido.
                Sé directo, estilo coach de competición.
                """
                respuesta = generar_respuesta_llama(prompt)
                st.markdown("### 📋 Informe de Llama Coach:")
                st.write(respuesta)
        else:
            st.warning("No hay suficientes datos en la base de datos.")