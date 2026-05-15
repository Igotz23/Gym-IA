import json

import streamlit as st
import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq
import google.generativeai as genai

# 1. Configuración Inicial
load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Gym Performance Tracker", layout="wide")

# Estilo minimalista para evitar distracciones
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    div.stButton > button:first-child { border-radius: 5px; border: 1px solid #3e444b; background-color: #1f2937; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE APOYO ---
def obtener_ultimo_entreno(grupo_muscular):
    try:
        response = supabase.table("registros_gym").select("*").eq("grupo_muscular", grupo_muscular).order("fecha", desc=True).limit(15).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            ultima_fecha = df['fecha'].max()
            ultimos = df[df['fecha'] == ultima_fecha]
            return [{"nombre": f['ejercicio'].upper(), "series": f['datos_series']} for _, f in ultimos.iterrows()]
    except: pass
    return []

def calcular_1rm_func(peso, reps):
    if reps <= 0: return 0
    return peso / (1.0278 - (0.0278 * reps)) if reps > 1 else peso

def generar_respuesta_llama(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error de conexión: {e}"
    
def analizar_plato_gemini(foto_archivo):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        image_data = foto_archivo.getvalue()
        
        # Formato correcto para Gemini 1.5
        image_parts = [
            {"mime_type": "image/jpeg", "data": image_data}
        ]

        prompt = """
        Analiza esta imagen de comida. Estima las cantidades y devuelve EXCLUSIVAMENTE 
        un objeto JSON con este formato:
        {"alimento": "nombre del plato", "calorias": 0, "proteinas": 0, "carbohidratos": 0, "grasas": 0}
        No escribas nada más que el JSON, sin bloques de código ni explicaciones.
        """

        response = model.generate_content([prompt, image_parts[0]])
        
        # LIMPIEZA AVANZADA: Quitamos posibles bloques de código Markdown
        texto = response.text
        if "```json" in texto:
            texto = texto.split("```json")[1].split("```")[0]
        elif "```" in texto:
            texto = texto.split("```")[1].split("```")[0]
        
        return texto.strip()
    except Exception as e:
        return f"Error con Gemini: {str(e)}"

# --- ESTRUCTURA DE PESTAÑAS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["REGISTRAR", "HISTORIAL", "EVOLUCIÓN", "COACH IA", "MACROS"])

# --- PESTAÑA 1: REGISTRO (CON PERSISTENCIA) ---
with tab1:
    st.title("Registro de Sesión")
    
    # Inicialización de estado de sesión para evitar pérdida de datos
    if 'lista_ejercicios' not in st.session_state:
        st.session_state.lista_ejercicios = []
    
    if st.session_state.lista_ejercicios:
        st.warning("Sesión en curso. Los datos se mantienen mientras la pestaña esté abierta.")

    col_fecha, col_grupo = st.columns(2)
    with col_fecha:
        fecha_entreno = st.date_input("Fecha", datetime.now())
    with col_grupo:
        grupo = st.selectbox("Distribución Muscular", 
                            ["Espalda + Bíceps", "Pecho + Hombro + Tríceps", "Pierna", "Hombro + Tríceps + Bíceps", "Espalda + Pecho"])

    c1, c2, c3 = st.columns(3)
    if c1.button("Importar pesos anteriores", use_container_width=True):
        st.session_state.lista_ejercicios = obtener_ultimo_entreno(grupo)
        st.rerun()
    
    if c2.button("Agregar ejercicio nuevo", use_container_width=True):
        st.session_state.lista_ejercicios.append({"nombre": "", "series": [{"peso": 0.0, "repes": 0}]})
        st.rerun()

    if c3.button("Limpiar sesión actual", type="secondary", use_container_width=True):
        st.session_state.lista_ejercicios = []
        st.rerun()

    # Renderizado de la lista de ejercicios
    for i, ej in enumerate(st.session_state.lista_ejercicios):
        with st.expander(f"EJERCICIO: {ej['nombre'] if ej['nombre'] else '...'}", expanded=True):
            # Guardado inmediato del nombre
            nombre_act = st.text_input("Nombre", value=ej['nombre'], key=f"n_{i}").upper().strip()
            st.session_state.lista_ejercicios[i]['nombre'] = nombre_act
            
            s_col1, s_col2 = st.columns(2)
            if s_col1.button(f"Añadir Serie", key=f"as_{i}"):
                st.session_state.lista_ejercicios[i]['series'].append({
                    "peso": ej['series'][-1]['peso'], 
                    "repes": ej['series'][-1]['repes']
                })
                st.rerun()

            if s_col2.button(f"Quitar Serie", key=f"rs_{i}") and len(ej['series']) > 1:
                st.session_state.lista_ejercicios[i]['series'].pop()
                st.rerun()
            
            for j, serie in enumerate(ej['series']):
                cx, cy, cz, cw = st.columns([0.5, 2, 2, 2])
                cx.markdown(f"<br>S{j+1}", unsafe_allow_html=True)
                
                # Sincronización inmediata de cada input con st.session_state
                p_val = cy.number_input("kg", key=f"p_{i}_{j}", value=float(serie['peso']), step=0.5)
                r_val = cz.number_input("reps", key=f"r_{i}_{j}", value=int(serie['repes']), step=1)
                
                st.session_state.lista_ejercicios[i]['series'][j]['peso'] = p_val
                st.session_state.lista_ejercicios[i]['series'][j]['repes'] = r_val
                
                cw.metric("1RM Est.", f"{round(calcular_1rm_func(p_val, r_val), 1)} kg")

    st.divider()
    if st.button("FINALIZAR Y GUARDAR EN NUBE", type="primary", use_container_width=True):
        if st.session_state.lista_ejercicios and st.session_state.lista_ejercicios[0]['nombre'] != "":
            datos = [{"fecha": str(fecha_entreno), "grupo_muscular": grupo, "ejercicio": e['nombre'], "datos_series": e['series']} for e in st.session_state.lista_ejercicios]
            supabase.table("registros_gym").insert(datos).execute()
            st.success("Sesión almacenada correctamente")
            st.balloons()
            st.session_state.lista_ejercicios = [] # Limpieza solo tras guardar con éxito
            st.rerun()
        else:
            st.error("No hay datos válidos para guardar")

# --- PESTAÑA 2: HISTORIAL ---
with tab2:
    st.title("Historial de Entrenamientos")
    fecha_busqueda = st.date_input("Consultar fecha:", datetime.now())
    res_historial = supabase.table("registros_gym").select("*").eq("fecha", str(fecha_busqueda)).execute()
    
    if res_historial.data:
        df_hist = pd.DataFrame(res_historial.data)
        st.info(f"Sesión: {df_hist['grupo_muscular'].iloc[0]}")
        for _, fila in df_hist.iterrows():
            st.subheader(fila['ejercicio'].upper())
            cols = st.columns(len(fila['datos_series']))
            for idx, s in enumerate(fila['datos_series']):
                cols[idx].metric(f"Serie {idx+1}", f"{s['peso']}kg", f"{s['repes']} reps", delta_color="off")
            st.divider()
    else:
        st.write("No hay registros para esta fecha.")

# --- PESTAÑA 3: EVOLUCIÓN ---
with tab3:
    st.title("Análisis de Progreso")
    res_graf = supabase.table("registros_gym").select("*").order("fecha").execute()
    
    if res_graf.data:
        df_todo = pd.DataFrame(res_graf.data)
        df_todo['ejercicio'] = df_todo['ejercicio'].str.upper()

        col_a, col_b = st.columns(2)
        with col_a:
            grupos = sorted(df_todo['grupo_muscular'].unique())
            sel_grupo = st.selectbox("Filtrar por grupo:", grupos)
        with col_b:
            ejercicios = sorted(df_todo[df_todo['grupo_muscular'] == sel_grupo]['ejercicio'].unique())
            sel_ej = st.selectbox("Seleccionar ejercicio:", ejercicios)
        
        df_ejercicio = df_todo[df_todo['ejercicio'] == sel_ej].copy()
        df_ejercicio['fecha'] = pd.to_datetime(df_ejercicio['fecha'])
        
        def procesar_volumen(lista_series):
            if not lista_series: return 0.0, 0
            pesos = [float(s['peso']) for s in lista_series]
            repes = [int(s['repes']) for s in lista_series]
            return max(pesos), sum(repes) # Suma total de repeticiones de todas las series
        
        df_ejercicio[['peso_max', 'repes_totales']] = df_ejercicio['datos_series'].apply(lambda x: pd.Series(procesar_volumen(x)))
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df_ejercicio['fecha'], y=df_ejercicio['peso_max'], name="Carga Máxima (kg)", marker_color='#4a90e2'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_ejercicio['fecha'], y=df_ejercicio['repes_totales'], name="Volumen Total (Reps)", line=dict(color='#2ecc71', width=3)), secondary_y=True)
        
        fig.update_layout(template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Datos insuficientes para generar gráficas.")

# --- PESTAÑA 4: COACH IA ---
with tab4:
    st.title("Asistente de Rendimiento")
    if st.button("Generar Informe Técnico", use_container_width=True):
        res = supabase.table("registros_gym").select("*").order("fecha", desc=True).limit(30).execute()
        if res.data:
            with st.spinner("Analizando métricas con Llama 3..."):
                prompt = f"Analiza estos datos de entrenamiento: {str(res.data)}. Da consejos técnicos muy directos y breves (máximo 3 puntos)."
                st.info(generar_respuesta_llama(prompt))
        else:
            st.warning("No hay suficientes datos registrados.")

# --- PESTAÑA 5: MACROS CON GEMINI ---
with tab5:
    st.title("Analizador de Comida")
    if 'm_temp' not in st.session_state:
        st.session_state.m_temp = {"al": "", "k": 0.0, "p": 0.0, "c": 0.0, "g": 0.0}

    foto = st.camera_input("Saca foto a tu plato")
    if foto and st.button("🔍 ANALIZAR CON GEMINI"):
        with st.spinner("Leyendo plato..."):
            res = analizar_plato_gemini(foto)
            # Si la función devuelve un mensaje de Error, lo mostramos
            if "Error" in res:
                st.error(res)
            else:
                try:
                    d = json.loads(res)
                    st.session_state.m_temp = {
                        "al": d.get('alimento', ''), 
                        "k": d.get('calorias', 0), 
                        "p": d.get('proteinas', 0), 
                        "c": d.get('carbohidratos', 0), 
                        "g": d.get('grasas', 0)
                    }
                    st.rerun() # Forzamos recarga para ver los datos en los inputs
                except Exception as e: 
                    st.error(f"La IA devolvió un formato extraño: {res}")

    st.divider()
    c1, c2 = st.columns(2)
    al = c1.text_input("Alimento", value=st.session_state.m_temp["al"]).upper()
    kcal = c1.number_input("kcal", value=float(st.session_state.m_temp["k"]))
    p = c2.number_input("Proteína (g)", value=float(st.session_state.m_temp["p"]))
    c = c2.number_input("Carbs (g)", value=float(st.session_state.m_temp["c"]))
    g = c2.number_input("Grasas (g)", value=float(st.session_state.m_temp["g"]))

    if st.button("🍎 GUARDAR MACROS", type="primary"):
        supabase.table("nutricion_gym").insert({"alimento": al, "calorias": kcal, "proteinas": p, "carbohidratos": c, "grasas": g}).execute()
        st.session_state.m_temp = {"al": "", "k": 0.0, "p": 0.0, "c": 0.0, "g": 0.0}
        st.success("Nutrición registrada"); st.rerun()

# (Las pestañas de HISTORIAL y COACH IA se mantienen con tu lógica anterior)