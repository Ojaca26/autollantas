# graph_builder.py

import pandas as pd
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, END

# Asegúrate de que tienes un archivo tools.py con todas estas funciones
from tools import (
    ejecutar_sql_real,
    analizar_con_datos,
    responder_conversacion,
    extraer_detalles_correo,
    enviar_correo_agente,
    generar_resumen_tabla,
    get_history_text,
    clasificar_intencion,
    validar_y_corregir_respuesta_analista # Importación añadida para completitud
)

# ===============================================================
# 1. DEFINICIÓN DEL ESTADO DEL GRAFO (LA MEMORIA COMPARTIDA)
# ===============================================================
class AgentState(TypedDict):
    """
    Define la estructura de la memoria del grafo que se comparte entre todos los nodos.
    Cada campo almacena una pieza de información a medida que el agente trabaja.
    """
    pregunta_usuario: str
    historial_chat: list
    clasificacion: str
    df: Optional[pd.DataFrame]
    analisis: Optional[str]
    respuesta_final: Optional[str]
    error: Optional[str]

# ===============================================================
# 2. DEFINICIÓN DE LOS NODOS DEL GRAFO (LOS AGENTES TRABAJADORES)
# ===============================================================
# Cada nodo es una función que realiza una tarea específica.
# Recibe el estado actual y devuelve un diccionario con los campos del estado que actualiza.

def nodo_clasificador(state: AgentState, llm_orq: ChatOpenAI):
    """Clasifica la intención del usuario para decidir la ruta inicial."""
    print("--- 🧠 NODO: Clasificador de Intención ---")
    pregunta = state["pregunta_usuario"]
    clasificacion = clasificar_intencion(pregunta, llm_orq)
    return {"clasificacion": clasificacion}

def nodo_consulta_sql(state: AgentState, llm_sql: ChatOpenAI, db: SQLDatabase):
    """Ejecuta la consulta SQL contra la base de datos y obtiene un DataFrame."""
    print("--- 🗃️ NODO: Consulta SQL ---")
    pregunta = state["pregunta_usuario"]
    hist_text = get_history_text(state["historial_chat"])
    
    # Llama a la función que traduce lenguaje natural a SQL y la ejecuta
    res_datos = ejecutar_sql_real(pregunta, hist_text, llm_sql, db)
    
    if res_datos.get("df") is not None and not res_datos["df"].empty:
        return {"df": res_datos["df"]}
    else:
        # Si no se obtienen datos, se registra un error para detener el flujo.
        return {"error": "No pude obtener datos para tu pregunta. Intenta reformularla."}

def nodo_analista(state: AgentState, llm_analista: ChatOpenAI):
    """Genera un análisis experto en texto basado en los datos del DataFrame."""
    print("--- 📊 NODO: Analista de Datos ---")
    pregunta = state["pregunta_usuario"]
    hist_text = get_history_text(state["historial_chat"])
    df = state["df"]
    
    analisis_texto = analizar_con_datos(pregunta, hist_text, df, llm_analista)
    return {"analisis": analisis_texto}

def nodo_validador(state: AgentState):
    """Valida la calidad del análisis. (Actualmente es un paso directo)."""
    print("--- 🕵️‍♀️ NODO: Supervisor de Calidad / Validador ---")
    # En una implementación futura, este nodo podría llamar a tu función `validar_y_corregir...`
    # y crear un bucle para reenviar la tarea al analista si es rechazada.
    # Por ahora, simplemente pasa el análisis como respuesta final.
    return {"respuesta_final": state["analisis"]}

def nodo_resumen_tabla(state: AgentState, llm_analista: ChatOpenAI):
    """Genera una introducción amable y conversacional para una tabla de datos."""
    print("--- ✍️ NODO: Resumen de Tabla ---")
    pregunta = state["pregunta_usuario"]
    res_datos = {"df": state["df"]} # Simula la estructura de entrada esperada
    
    res_con_intro = generar_resumen_tabla(pregunta, res_datos, llm_analista)
    return {"respuesta_final": res_con_intro.get("texto")}

def nodo_conversacional(state: AgentState, llm_analista: ChatOpenAI):
    """Maneja saludos y preguntas generales que no requieren datos."""
    print("--- 👋 NODO: Conversacional ---")
    pregunta = state["pregunta_usuario"]
    hist_text = get_history_text(state["historial_chat"])
    
    res = responder_conversacion(pregunta, hist_text, llm_analista)
    return {"respuesta_final": res.get("texto")}
    
def nodo_correo(state: AgentState, llm_analista: ChatOpenAI):
    """Extrae detalles de la solicitud y envía un correo con los datos más recientes."""
    print("--- 📧 NODO: Agente de Correo ---")
    pregunta = state["pregunta_usuario"]
    historial = state["historial_chat"]
    df_para_enviar = None

    # Busca el último DataFrame en el historial del chat para adjuntarlo
    for msg in reversed(historial):
        if msg.get('role') == 'assistant':
            content = msg.get('content', {})
            df_prev = content.get('df')
            if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                df_para_enviar = df_prev
                break

    detalles = extraer_detalles_correo(pregunta, llm_analista)
    resultado_envio = enviar_correo_agente(
        recipient=detalles["recipient"],
        subject=detalles["subject"],
        body=detalles["body"],
        df=df_para_enviar
    )
    return {"respuesta_final": resultado_envio.get("texto")}

# ===============================================================
# 3. DEFINICIÓN DE LAS ARISTAS (LAS DECISIONES DEL GRAFO)
# ===============================================================
# Las aristas son funciones que dirigen el flujo de un nodo a otro.

def decidir_ruta(state: AgentState):
    """Lee la clasificación del primer nodo y decide la ruta principal."""
    print(f"--- 🧭 DECISIÓN: Ruta basada en clasificación '{state['clasificacion']}' ---")
    if state.get("error"):
        return "end" # Si hay un error, el flujo termina.
        
    clasificacion = state["clasificacion"]
    if clasificacion == "conversacional":
        return "conversacional"
    elif clasificacion == "correo":
        return "correo"
    elif clasificacion in ["analista", "consulta"]:
        # Tanto para consultas como para análisis, el primer paso es obtener datos.
        return "consulta_sql"
    else:
        # Si la clasificación no es clara, termina el flujo.
        return "end"

def decidir_despues_sql(state: AgentState):
    """Una vez obtenidos los datos, decide si se necesita un análisis o solo un resumen."""
    print("--- 🧭 DECISIÓN: Ruta después de la consulta SQL ---")
    if state.get("error"):
        return "end"
        
    if state["clasificacion"] == "analista":
        # Si la intención original era analizar, pasa al nodo analista.
        return "analista"
    else: 
        # Si era una consulta directa, solo resume la tabla.
        return "resumen_tabla"

# ===============================================================
# 4. CONSTRUCCIÓN Y COMPILACIÓN DEL GRAFO
# ===============================================================
def create_graph(llms: dict, db: SQLDatabase):
    """
    Esta función construye el diagrama de flujo ejecutable:
    1. Define los nodos.
    2. Establece el punto de entrada.
    3. Conecta los nodos con las aristas (las reglas de decisión).
    4. Compila todo en una aplicación lista para ser usada.
    """
    
    # "Bind" es una técnica para pre-configurar los nodos con las herramientas
    # que necesitan (los LLMs y la conexión a la BD), simplificando su llamada.
    bound_nodo_clasificador = lambda state: nodo_clasificador(state, llms["orq"])
    bound_nodo_sql = lambda state: nodo_consulta_sql(state, llms["sql"], db)
    bound_nodo_analista = lambda state: nodo_analista(state, llms["analista"])
    bound_nodo_validador = nodo_validador
    bound_nodo_resumen = lambda state: nodo_resumen_tabla(state, llms["analista"])
    bound_nodo_conversacional = lambda state: nodo_conversacional(state, llms["analista"])
    bound_nodo_correo = lambda state: nodo_correo(state, llms["analista"])
    
    workflow = StateGraph(AgentState)

    # Añadir todos los nodos ("cajas") al diagrama de flujo
    workflow.add_node("clasificador", bound_nodo_clasificador)
    workflow.add_node("consulta_sql", bound_nodo_sql)
    workflow.add_node("analista", bound_nodo_analista)
    workflow.add_node("validador", bound_nodo_validador)
    workflow.add_node("resumen_tabla", bound_nodo_resumen)
    workflow.add_node("conversacional", bound_nodo_conversacional)
    workflow.add_node("correo", bound_nodo_correo)

    # Definir por dónde empieza siempre el flujo
    workflow.set_entry_point("clasificador")

    # Añadir las aristas condicionales ("rombos de decisión")
    workflow.add_conditional_edges(
        "clasificador",
        decidir_ruta,
        {
            "conversacional": "conversacional",
            "correo": "correo",
            "consulta_sql": "consulta_sql",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "consulta_sql",
        decidir_despues_sql,
        {
            "analista": "analista",
            "resumen_tabla": "resumen_tabla",
            "end": END
        }
    )

    # Añadir las aristas normales ("flechas directas")
    workflow.add_edge("analista", "validador")
    workflow.add_edge("validador", END)
    workflow.add_edge("resumen_tabla", END)
    workflow.add_edge("conversacional", END)
    workflow.add_edge("correo", END)

    # Compilar el grafo en una aplicación ejecutable
    app = workflow.compile()

    return app
