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
st.set_page_config(page_title="IANA con LangGraph", page_icon="logo_autollantas.png", layout="wide")

col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo_autollantas.png", width=120)
with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **AutoLLantas**. Pregúntame algo sobre tu negocio.")

# ============================================
# 1) Conexión a BD, LLMs y Grafo
# ============================================

# Usamos st.session_state para una inicialización más controlada
def initialize_connections():
    if "db_connection_status" not in st.session_state:
        st.session_state.db_connection_status = "pending"
        st.session_state.db = None
        st.session_state.llms = None
        st.session_state.graph = None

    if st.session_state.db_connection_status == "pending":
        try:
            with st.spinner("🛰️ Conectando a la base de datos..."):
                creds = st.secrets["db_credentials"]
                uri = f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
                engine_args = {"connect_args": {"ssl_disabled": True}}
                db = SQLDatabase.from_uri(uri, include_tables=["autollantas"], engine_args=engine_args)
                db.run("SELECT 1")
                st.session_state.db = db
                st.success("✅ Conexión a la base de datos establecida.")
                st.session_state.db_connection_status = "success"
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            st.session_state.db_connection_status = "failed"
            return

    if st.session_state.db_connection_status == "success" and not st.session_state.llms:
        try:
            with st.spinner("🤝 Inicializando la red de agentes IANA..."):
                api_key = st.secrets["openai_api_key"]
                model_name = "gpt-4o"
                st.session_state.llms = {
                    "sql": ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=api_key),
                    "analista": ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=api_key),
                    "orq": ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key),
                    "validador": ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key),
                }
                st.success("✅ Agentes de IANA listos.")
        except Exception as e:
            st.error(f"Error al inicializar los LLMs: {e}")
            return

    if st.session_state.db and st.session_state.llms and not st.session_state.graph:
        with st.spinner("🕸️ Construyendo la red de agentes..."):
            st.session_state.graph = create_graph(st.session_state.llms, st.session_state.db)
            st.success("✅ Red de agentes IANA compilada con LangGraph.")

# Llama a la función de inicialización al principio del script
initialize_connections()

# ============================================
# 2) Interfaz: Chat y Procesamiento de Preguntas
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": {"texto": "¡Hola! Soy IANA. ¿Qué te gustaría analizar hoy?"}}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message.get("content", {})
        if isinstance(content, dict):
            if content.get("texto"): st.markdown(content["texto"])
            if isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty: st.dataframe(content["df"])
            if content.get("analisis"): st.markdown(content["analisis"])
        elif isinstance(content, str):
            st.markdown(content)

def procesar_pregunta(prompt: str):
    # La comprobación ahora se basa en st.session_state.graph
    if not st.session_state.graph:
        st.error("La red de agentes no está inicializada. Revisa los mensajes de error de conexión o de API keys más arriba.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            initial_state = {
                "pregunta_usuario": prompt,
                "historial_chat": st.session_state.messages
            }
            final_state = st.session_state.graph.invoke(initial_state)

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
                if final_state.get("clasificacion") == "analista" and final_state.get("analisis"):
                     response_content["analisis"] = final_state["analisis"]

            st.session_state.messages.append({"role": "assistant", "content": response_content})

prompt = st.chat_input("Escribe tu pregunta aquí...")
if prompt:
    procesar_pregunta(prompt)
