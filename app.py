# app.py

import streamlit as st
import pandas as pd
from typing import Optional

# LangChain + Conexión
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

# Importa el constructor del grafo
from graph_builder import create_graph

# ============================================
# 0) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="IANA con LangGraph", page_icon="logo_automundial.png", layout="wide")

col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo_automundial.png", width=120)
with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **Automundial**. Pregúntame algo sobre tu negocio.")

# ============================================
# 1) Conexión a BD, LLMs y Grafo (Manejo de Cache)
# ============================================

@st.cache_resource
def get_database_connection():
    # ... (tu código de conexión original, sin cambios)
    pass # Reemplaza con tu función original

@st.cache_resource
def get_llms():
    # ... (tu código original para inicializar LLMs)
    # MODIFICACIÓN: Devolver un diccionario para mayor claridad
    try:
        api_key = st.secrets["openai_api_key"]
        model_name = "gpt-4o"
        llms = {
            "sql": ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=api_key),
            "analista": ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=api_key),
            "orq": ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key),
            "validador": ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key),
        }
        st.success("✅ Agentes de IANA listos.")
        return llms
    except Exception as e:
        st.error(f"Error al inicializar los LLMs: {e}")
        return None

db = get_database_connection()
llms = get_llms()

@st.cache_resource
def get_compiled_graph(_llms, _db):
    if not _llms or not _db:
        return None
    with st.spinner("🕸️ Construyendo la red neuronal de agentes..."):
        graph = create_graph(_llms, _db)
        st.success("✅ Red de agentes IANA compilada con LangGraph.")
        return graph

graph = get_compiled_graph(llms, db)

# ============================================
# 2) Interfaz: Chat y Procesamiento de Preguntas
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": {"texto": "¡Hola! Soy IANA. ¿Qué te gustaría analizar hoy?"}}]

# Muestra el historial del chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message.get("content", {})
        if isinstance(content, dict):
            if content.get("texto"): st.markdown(content["texto"])
            if isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty: st.dataframe(content["df"])
            if content.get("analisis"): st.markdown(content["analisis"])
        elif isinstance(content, str):
             st.markdown(content) # Para las preguntas del usuario

# Función principal que ahora llama al grafo
def procesar_pregunta(prompt: str):
    if not graph:
        st.error("La red de agentes no está inicializada. Revisa la conexión a la base de datos o las API keys.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Define el estado inicial para el grafo
            initial_state = {
                "pregunta_usuario": prompt,
                "historial_chat": st.session_state.messages
            }
            
            # ¡Aquí es donde se ejecuta el grafo!
            final_state = graph.invoke(initial_state)

            # Prepara la respuesta para mostrarla y guardarla
            response_content = {}
            if final_state.get("error"):
                response_content["texto"] = f"❌ Lo siento, ocurrió un error: {final_state['error']}"
                st.error(response_content["texto"])
            else:
                if final_state.get("respuesta_final"):
                    response_content["texto"] = final_state["respuesta_final"]
                    st.markdown(response_content["texto"])
                
                if final_state.get("df") is not None and not final_state["df"].empty:
                    response_content["df"] = final_state["df"]
                    st.dataframe(response_content["df"])
                
                # Si el análisis es la respuesta final, también lo guardamos
                if final_state.get("clasificacion") == "analista" and final_state.get("analisis"):
                     response_content["analisis"] = final_state.get("analisis")

            st.session_state.messages.append({"role": "assistant", "content": response_content})


# Input del usuario (texto o voz) - sin cambios
prompt = st.chat_input("Escribe tu pregunta aquí...")
if prompt:
    procesar_pregunta(prompt)