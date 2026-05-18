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
        response = supabase.table("registros_gym").select("*").eq("grupo_muscular", grupo_muscular).order("fecha", desc=True).limit(20).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            ultima_fecha = df['fecha'].max()
            ultimos = df[df['fecha'] == ultima_fecha].copy()

            # --- APLICAMOS TU ORDEN AQUÍ TAMBIÉN ---
            def asignar_prioridad_local(nombre_ej):
                nombre = nombre_ej.upper()
                if any(x in nombre for x in ["JALÓN", "REMO", "DOMINADAS", "PESO MUERTO", "PRESS", "SENTADILLA", "PRENSA", "PECK DECK", "APERTURAS"]):
                    return 1
                if any(x in nombre for x in ["ELEVACIONES", "MILITAR", "POSTERIOR", "LATERALES"]):
                    return 2
                if any(x in nombre for x in ["TRÍCEPS", "EXTENSION"]):
                    return 3
                if any(x in nombre for x in ["CURL","BÍCEPS", "MARTILLO", "BAYESIAN"]):
                    return 4
                return 5

            ultimos['prio'] = ultimos['ejercicio'].apply(asignar_prioridad_local)
            ultimos = ultimos.sort_values(by='prio')
            
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
        # Usamos el modelo actualizado y estable
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        image_data = foto_archivo.getvalue()
        image_parts = [{"mime_type": "image/jpeg", "data": image_data}]

        prompt = """
      Analiza esta imagen de comida con extrema precisión, actuando como un nutricionista clínico experto en estimación visual de porciones.
        
        INSTRUCCIONES CRÍTICAS DE ESTIMACIÓN:
        1. Identifica los ingredientes principales visiblemente presentes.
        2. Estima el VOLUMEN en ml o cm³ de la comida basándote en la referencia del recipiente (tupper, plato) y los cubiertos. No asumas raciones de restaurante.
        3. Aplica densidad nutricional estándar para alimentos cocidos. Por ejemplo: para pasta cocida, estima la cantidad que cabe en ese volumen y calcula sus macros específicos (aprox. 1.3g carbs por g de pasta cocida).
        4. Si hay salsa, estima su volumen y composición por separado (ej. salsa de tomate: base agua, bajos macros vs salsa carbonara: alta grasa).
        5. SUMA las estimaciones de todos los ingredientes para el total del plato.

        FORMATO DE SALIDA (EXCLUSIVAMENTE JSON Raw, sin bloques de código ```):
        {
          "alimento": "nombre descriptivo y corto del plato",
          "calorias": 0.0,
          "proteinas": 0.0,
          "carbohidratos": 0.0,
          "grasas": 0.0
        }

        Regla de oro: Es preferible una estimación conservadora basada en el volumen visible que una estimación inflada basada en 'raciones tipo'.
        """

        response = model.generate_content([prompt, image_parts[0]])
        
        # Limpieza por si acaso devuelve bloques markdown
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
            
# --- PESTAÑA 2: HISTORIAL (CORREGIDA Y ORDENADA) ---
with tab2:
    st.title("Historial de Entrenamientos")
    fecha_busqueda = st.date_input("Consultar fecha:", datetime.now())
    res_historial = supabase.table("registros_gym").select("*").eq("fecha", str(fecha_busqueda)).execute()
    
    if res_historial.data:
        df_hist = pd.DataFrame(res_historial.data)
        
        # 1. FUNCIÓN DE PRIORIDAD (Tu lógica actualizada)
        def asignar_prioridad(nombre_ej):
            nombre = nombre_ej.upper()
            if any(x in nombre for x in ["PRESS","PECK DECK", "APERTURAS","DOMINADAS","JALÓN", "REMO","PESO MUERTO","SENTADILLA", "PRENSA"]):
                return 1
            if any(x in nombre for x in ["ELEVACIONES", "MILITAR", "POSTERIOR", "LATERALES"]):
                return 2
            if any(x in nombre for x in ["FONDOS","TRÍCEPS", "EXTENSION"]):
                return 3
            if any(x in nombre for x in ["PREDICADOR","CURL","BÍCEPS", "MARTILLO", "BAYESIAN"]):
                return 4
            return 5

        # IMPORTANTE: Aplicamos el orden al DataFrame antes de CUALQUIER otra cosa
        df_hist['prioridad'] = df_hist['ejercicio'].apply(asignar_prioridad)
        df_hist = df_hist.sort_values(by=['prioridad', 'ejercicio']) # Ordena por prioridad y luego alfabético

        # 2. CONSTRUIR EL TEXTO DE COPIA (ESTILO NOTAS)
        grupo_nombre = df_hist['grupo_muscular'].iloc[0]
        fecha_txt = fecha_busqueda.strftime("%d/%m/%y")
        resumen_txt = f"{grupo_nombre} — {fecha_txt}\n"
        
        ultima_prioridad = 0
        for _, fila in df_hist.iterrows():
            prio_actual = fila['prioridad']
            # Añadimos cabeceras de sección al texto de copia
            if prio_actual != ultima_prioridad:
                if prio_actual == 1: resumen_txt += "\n--- BLOQUE PRINCIPAL ---\n"
                elif prio_actual == 2: resumen_txt += "\n--- HOMBRO ---\n"
                elif prio_actual == 3: resumen_txt += "\n--- TRÍCEPS ---\n"
                elif prio_actual == 4: resumen_txt += "\n--- BÍCEPS ---\n"
                ultima_prioridad = prio_actual

            ej_nombre = fila['ejercicio'].upper()
            series = fila['datos_series']
            peso_ref = series[0]['peso']
            
            resumen_txt += f"\n{ej_nombre} ({peso_ref} kg)\n"
            repes_linea = " - ".join([str(s['repes']) for s in series])
            resumen_txt += f"{repes_linea}\n"

        # 3. MOSTRAR INTERFAZ
        st.subheader("📋 Resumen para Notas")
        st.code(resumen_txt, language="text")
        
        st.divider()
        
        # 4. VISTA DETALLADA (Ahora también saldrá ordenada)
        st.subheader("Detalle del Entrenamiento")
        for _, fila in df_hist.iterrows():
            with st.container():
                st.markdown(f"### {fila['ejercicio'].upper()}")
                cols = st.columns(len(fila['datos_series']))
                for idx, s in enumerate(fila['datos_series']):
                    cols[idx].metric(f"Serie {idx+1}", f"{s['peso']}kg", f"{s['repes']} reps")
                st.write("") # Espacio entre ejercicios
    else:
        st.info("No hay registros para este día.")
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
# --- PESTAÑA 4: COACH IA (REPARADO Y ULTRA-PRECISO) ---
with tab4:
    st.title("Asistente de Rendimiento Pro")
    if st.button("📊 Analizar Sobrecarga Progresiva", use_container_width=True):
        # Traemos los últimos 100 registros
        res = supabase.table("registros_gym").select("*").order("fecha", desc=True).limit(100).execute()
        
        if res.data:
            with st.spinner("Analizando la tendencia de tus cargas..."):
                
                # --- LIMPIEZA DE DATOS PREVIA ---
                # Pasamos solo la información limpia para que la IA no se confunda con IDs técnicos
                datos_limpios = []
                for r in res.data:
                    datos_limpios.append({
                        "fecha": r.get("fecha"),
                        "ejercicio": r.get("ejercicio", "").upper(),
                        "series_y_repes": r.get("datos_series")
                    })
                
                # El prompt ahora es ultra-estricto con la cronología y el formato
                prompt = f"""
                Actúa como un entrenador de fuerza de élite, experto en sobrecarga progresiva.
                Analiza esta lista de entrenamientos ordenados cronológicamente desde el más reciente al más antiguo: {str(datos_limpios)}.
                
                CRÍTICO PARA EL ANÁLISIS:
                - La primera aparición de un ejercicio en la lista es su nivel ACTUAL (lo más reciente). Compara esto con sus apariciones posteriores (pasadas).
                - Si en la última sesión el usuario subió los kg o las repeticiones respecto a las anteriores, eso es PROGRESO, jamás lo listes como estancamiento.
                - Solo hay estancamiento si en las últimas 3 o 4 veces que hizo el ejercicio, los kg y las repeticiones se mantuvieron exactamente iguales.

                REGLAS DE FORMATO:
                - Prohibido usar IDs, números de ejercicio, contraseñas o textos técnicos de bases de datos.
                - Nombra los ejercicios tal cual los escribe el usuario.
                - Ve al grano, usa frases cortas y motivadoras.

                Estructura tu respuesta EXACTAMENTE así:

                🚨 **ALERTAS DE ESTANCAMIENTO (¡A apretar!):**
                - [Nombre del ejercicio]: Breve motivo real de por qué lleva congelado varias sesiones.

                🔥 **OPORTUNIDADES DE SUBIR CARGA (A por ello):**
                - [Nombre del ejercicio]: Breve motivo de por qué sus últimas series indican que ya tolera más peso.

                💡 **CONSEJO TÁCTICO CORTO:**
                - Un consejo práctico de una sola línea adaptado a lo que ves en sus datos.
                """
                
                st.markdown("### 📋 Informe Técnico del Coach")
                st.info(generar_respuesta_llama(prompt))
        else:
            st.warning("No hay suficientes datos registrados en la nube para analizar tu progresión.")
# --- PESTAÑA 5: MACROS CON GEMINI, TEXTO Y OBJETIVOS (BLOQUE COMPLETO) ---
with tab5:
    st.title("Diario de Nutrición Inteligente")
    
    # 1. CONFIGURACIÓN DE OBJETIVOS (Se guardan en la sesión)
    if 'obj_kcal' not in st.session_state: st.session_state.obj_kcal = 2500.0
    if 'obj_p' not in st.session_state: st.session_state.obj_p = 150.0
    if 'obj_c' not in st.session_state: st.session_state.obj_c = 250.0
    if 'obj_g' not in st.session_state: st.session_state.obj_g = 70.0

    with st.expander("⚙️ Configurar Mis Objetivos Diarios"):
        c_obj1, c_obj2, c_obj3, c_obj4 = st.columns(4)
        st.session_state.obj_kcal = c_obj1.number_input("Objetivo Kcal", value=float(st.session_state.obj_kcal), step=50.0)
        st.session_state.obj_p = c_obj2.number_input("Objetivo Prot (g)", value=float(st.session_state.obj_p), step=5.0)
        st.session_state.obj_c = c_obj3.number_input("Objetivo Carbs (g)", value=float(st.session_state.obj_c), step=5.0)
        st.session_state.obj_g = c_obj4.number_input("Objetivo Grasas (g)", value=float(st.session_state.obj_g), step=5.0)

    st.divider()

    # 2. ENTRADA DE DATOS (FOTO O TEXTO MANUAL)
    if 'm_temp' not in st.session_state:
        st.session_state.m_temp = {"al": "", "k": 0.0, "p": 0.0, "c": 0.0, "g": 0.0}

    tipo_entrada = st.radio("Método de registro:", ["📷 Usar Foto", "✍️ Escribir a mano (IA)"], horizontal=True)

    if tipo_entrada == "📷 Usar Foto":
        metodo_foto = st.radio("Origen de la imagen:", ["Cámara", "Galería"], horizontal=True)
        # Soportamos HEIC explícitamente en el file_uploader
        foto = st.camera_input("Saca foto") if metodo_foto == "Cámara" else st.file_uploader("Sube imagen", type=["jpg", "jpeg", "png", "heic"])
        
        if foto:
            st.image(foto, caption="Plato detectado", use_container_width=True)
            if st.button("🔍 ANALIZAR FOTO CON GEMINI", use_container_width=True):
                with st.spinner("Gemini analizando el plato con precisión..."):
                    res = analizar_plato_gemini(foto)
                    if "Error" in res: 
                        st.error(res)
                    else:
                        try:
                            # Limpieza ultra-robusta de marcas de formato de la IA
                            res_limpio = res.strip()
                            if res_limpio.startswith("```"):
                                res_limpio = res_limpio.split("\n", 1)[1].rsplit("\n", 1)[0]
                            if res_limpio.lower().startswith("json"):
                                res_limpio = res_limpio.split("json", 1)[1].strip()
                            
                            d = json.loads(res_limpio)
                            
                            # Forzamos float() para admitir decimales precisos sin romper la app
                            st.session_state.m_temp = {
                                "al": str(d.get('alimento', '')).upper(), 
                                "k": float(d.get('calorias', 0.0)), 
                                "p": float(d.get('proteinas', 0.0)), 
                                "c": float(d.get('carbohidratos', 0.0)), 
                                "g": float(d.get('grasas', 0.0))
                            }
                            st.rerun()
                        except Exception as e: 
                            st.error(f"Error al procesar el JSON. La IA devolvió: {res}")

    else:
        texto_comida = st.text_input("Ejemplo: '200gr de macarrones con tomate frito y 150gr de pechuga de pollo'")
        if texto_comida and st.button("🤖 ESTIMAR MACROS POR TEXTO", use_container_width=True):
            with st.spinner("Llama 3 calculando macros..."):
                prompt_texto = f"""
                Analiza textualmente esta comida: '{texto_comida}'. Estima las cantidades y devuelve EXCLUSIVAMENTE 
                un objeto JSON con este formato (valores numéricos aproximados):
                {{"alimento": "resumen corto de lo que comió", "calorias": 0, "proteinas": 0, "carbohidratos": 0, "grasas": 0}}
                No escribas absolutamente nada más que el JSON raw, sin bloques de código ```json ni texto adicional.
                """
                res = generar_respuesta_llama(prompt_texto)
                try:
                    res_limpio = res.replace('```json', '').replace('```', '').strip()
                    d = json.loads(res_limpio)
                    st.session_state.m_temp = {
                        "al": str(d.get('alimento', '')).upper(), 
                        "k": float(d.get('calorias', 0.0)), 
                        "p": float(d.get('proteinas', 0.0)), 
                        "c": float(d.get('carbohidratos', 0.0)), 
                        "g": float(d.get('grasas', 0.0))
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al procesar el texto. La IA respondió: {res}")

    # 3. FORMULARIO DE CONFIRMACIÓN ANTES DE GUARDAR
    st.subheader("Confirmar Macronutrientes")
    c1, c2 = st.columns(2)
    al = c1.text_input("Alimento", value=st.session_state.m_temp["al"]).upper()
    kcal = c1.number_input("kcal", value=float(st.session_state.m_temp["k"]))
    p = c2.number_input("Proteína (g)", value=float(st.session_state.m_temp["p"]))
    c = c2.number_input("Carbs (g)", value=float(st.session_state.m_temp["c"]))
    g = c2.number_input("Grasas (g)", value=float(st.session_state.m_temp["g"]))

    if st.button("🍎 GUARDAR EN DIARIO", type="primary", use_container_width=True):
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        supabase.table("nutricion_gym").insert({
            "alimento": al, "calorias": kcal, "proteinas": p, "carbohidratos": c, "grasas": g, "fecha": fecha_hoy
        }).execute()
        st.session_state.m_temp = {"al": "", "k": 0.0, "p": 0.0, "c": 0.0, "g": 0.0}
        st.success("Comida añadida al historial de hoy!")
        st.rerun()

    st.divider()

    # 4. DASHBOARD: TOTALES DEL DÍA VS OBJETIVOS
    st.subheader("📊 Progreso Diario de Hoy")
    fecha_hoy_str = datetime.now().strftime("%Y-%m-%d")
    
    # Traemos las comidas registradas en la fecha actual
    res_hoy = supabase.table("nutricion_gym").select("*").eq("fecha", fecha_hoy_str).execute()
    
    tot_kcal, tot_p, tot_c, tot_g = 0.0, 0.0, 0.0, 0.0
    
    if res_hoy.data:
        df_hoy = pd.DataFrame(res_hoy.data)
        tot_kcal = df_hoy['calorias'].sum()
        tot_p = df_hoy['proteinas'].sum()
        tot_c = df_hoy['carbohidratos'].sum()
        tot_g = df_hoy['grasas'].sum()

    # Funciones para calcular lo restante o excesos
    def txt_delta(actual, objetivo):
        dif = objetivo - actual
        return f"{round(dif, 1)}g restantes" if dif >= 0 else f"{round(abs(dif), 1)}g de exceso"

    def txt_delta_kcal(actual, objetivo):
        dif = objetivo - actual
        return f"{round(dif, 1)} kcal rest." if dif >= 0 else f"{round(abs(dif), 1)} kcal exc."

    # Renderizado en tarjetas de métricas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calorías Totales", f"{round(tot_kcal, 1)} / {st.session_state.obj_kcal}", txt_delta_kcal(tot_kcal, st.session_state.obj_kcal), delta_color="inverse")
    m2.metric("Proteínas", f"{round(tot_p, 1)}g / {st.session_state.obj_p}g", txt_delta(tot_p, st.session_state.obj_p))
    m3.metric("Carbohidratos", f"{round(tot_c, 1)}g / {st.session_state.obj_c}g", txt_delta(tot_c, st.session_state.obj_c))
    m4.metric("Grasas", f"{round(tot_g, 1)}g / {st.session_state.obj_g}g", txt_delta(tot_g, st.session_state.obj_g))

    # Barra visual de carga energética del día
    pct = min(tot_kcal / max(st.session_state.obj_kcal, 1.0), 1.0)
    st.progress(pct, text=f"Energía consumida: {round(pct*100, 1)}%")