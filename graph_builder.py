# graph_builder.py

import pandas as pd
from typing import TypedDict, Annotated, Optional
import operator

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, END

# Importa las funciones "trabajadoras" que ya tenías
# (Asumimos que las moviste a un archivo `tools.py` o las dejas en este archivo)
from tools import (
    ejecutar_sql_real,
    analizar_con_datos,
    validar_y_corregir_respuesta_analista,
    responder_conversacion,
    extraer_detalles_correo,
    enviar_correo_agente,
    generar_resumen_tabla,
    get_history_text
)

# ===============================================================
# 1. DEFINICIÓN DEL ESTADO DEL GRAFO (LA MEMORIA COMPARTIDA)
# ===============================================================
class AgentState(TypedDict):
    """
    Define la estructura de la memoria del grafo.
    - `add_operator`: Permite que los valores se acumulen en lugar de reemplazarse.
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
# Cada nodo es una función que recibe el estado actual y devuelve
# un diccionario para actualizar ese estado.

def nodo_clasificador(state: AgentState, llm_orq: ChatOpenAI):
    """Clasifica la intención del usuario para decidir la ruta."""
    print("--- 🧠 NODO: Clasificador de Intención ---")
    pregunta = state["pregunta_usuario"]
    # (Aquí iría la lógica de tu función `clasificar_intencion` original)
    # Por simplicidad, la llamaremos directamente desde `tools.py` si la moviste
    from tools import clasificar_intencion
    clasificacion = clasificar_intencion(pregunta, llm_orq)
    return {"clasificacion": clasificacion}

def nodo_consulta_sql(state: AgentState, llm_sql: ChatOpenAI, db: SQLDatabase):
    """Ejecuta la consulta SQL y obtiene un DataFrame."""
    print("--- 🗃️ NODO: Consulta SQL ---")
    pregunta = state["pregunta_usuario"]
    hist_text = get_history_text(state["historial_chat"])
    
    # Llama a tu función original para obtener los datos
    res_datos = ejecutar_sql_real(pregunta, hist_text, llm_sql, db)
    
    if res_datos.get("df") is not None and not res_datos["df"].empty:
        return {"df": res_datos["df"]}
    else:
        return {"error": "No pude obtener datos para tu pregunta. Intenta reformularla."}

def nodo_analista(state: AgentState, llm_analista: ChatOpenAI):
    """Genera un análisis experto basado en los datos del DataFrame."""
    print("--- 📊 NODO: Analista de Datos ---")
    pregunta = state["pregunta_usuario"]
    hist_text = get_history_text(state["historial_chat"])
    df = state["df"]
    
    analisis_texto = analizar_con_datos(pregunta, hist_text, df, llm_analista)
    return {"analisis": analisis_texto}

def nodo_validador(state: AgentState, llm_validador: ChatOpenAI):
    """Valida la calidad del análisis y decide si se necesita una corrección."""
    print("--- 🕵️‍♀️ NODO: Supervisor de Calidad / Validador ---")
    # En una implementación real, aquí iría la lógica de tu validador,
    # que podría devolver un feedback para re-ejecutar el nodo_analista.
    # Por ahora, simplemente aprobamos.
    return {"respuesta_final": state["analisis"]}

def nodo_resumen_tabla(state: AgentState, llm_analista: ChatOpenAI):
    """Genera una introducción conversacional para la tabla."""
    print("--- ✍️ NODO: Resumen de Tabla ---")
    pregunta = state["pregunta_usuario"]
    res_datos = {"df": state["df"]} # Simula la estructura que esperaba tu función
    
    res_con_intro = generar_resumen_tabla(pregunta, res_datos, llm_analista)
    return {"respuesta_final": res_con_intro.get("texto")}

def nodo_conversacional(state: AgentState, llm_analista: ChatOpenAI):
    """Maneja saludos y preguntas generales."""
    print("--- 👋 NODO: Conversacional ---")
    pregunta = state["pregunta_usuario"]
    hist_text = get_history_text(state["historial_chat"])
    
    res = responder_conversacion(pregunta, hist_text, llm_analista)
    return {"respuesta_final": res.get("texto")}
    
def nodo_correo(state: AgentState, llm_analista: ChatOpenAI):
    """Extrae detalles y envía un correo."""
    print("--- 📧 NODO: Agente de Correo ---")
    pregunta = state["pregunta_usuario"]
    historial = state["historial_chat"]
    df_para_enviar = None

    # Busca el último DataFrame en el historial
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
def decidir_ruta(state: AgentState):
    """Lee la clasificación y decide a qué nodo ir a continuación."""
    print(f"--- 🧭 DECISIÓN: Ruta basada en clasificación '{state['clasificacion']}' ---")
    if state["error"]:
        return "end" # Si hay un error, termina.
        
    clasificacion = state["clasificacion"]
    if clasificacion == "conversacional":
        return "conversacional"
    elif clasificacion == "correo":
        return "correo"
    elif clasificacion == "analista":
        return "consulta_sql" # El analista necesita datos primero
    elif clasificacion == "consulta":
        return "consulta_sql"
    else:
        return "end"

def decidir_despues_sql(state: AgentState):
    """Después de obtener los datos, decide si necesita análisis o solo un resumen."""
    print("--- 🧭 DECISIÓN: Ruta después de la consulta SQL ---")
    if state["error"]:
        return "end"
        
    if state["clasificacion"] == "analista":
        return "analista"
    else: # Es una consulta directa
        return "resumen_tabla"

# ===============================================================
# 4. CONSTRUCCIÓN Y COMPILACIÓN DEL GRAFO
# ===============================================================
def create_graph(llms: dict, db: SQLDatabase):
    """Crea y compila el grafo de LangGraph."""
    
    # Bind de los LLMs a los nodos para no pasarlos cada vez
    # Esto es una forma limpia de dar a cada nodo las herramientas que necesita
    bound_nodo_clasificador = lambda state: nodo_clasificador(state, llms["orq"])
    bound_nodo_sql = lambda state: nodo_consulta_sql(state, llms["sql"], db)
    bound_nodo_analista = lambda state: nodo_analista(state, llms["analista"])
    bound_nodo_validador = lambda state: nodo_validador(state, llms["validador"])
    bound_nodo_resumen = lambda state: nodo_resumen_tabla(state, llms["analista"])
    bound_nodo_conversacional = lambda state: nodo_conversacional(state, llms["analista"])
    bound_nodo_correo = lambda state: nodo_correo(state, llms["analista"])
    
    workflow = StateGraph(AgentState)

    # Añadir todos los nodos al grafo
    workflow.add_node("clasificador", bound_nodo_clasificador)
    workflow.add_node("consulta_sql", bound_nodo_sql)
    workflow.add_node("analista", bound_nodo_analista)
    workflow.add_node("validador", bound_nodo_validador)
    workflow.add_node("resumen_tabla", bound_nodo_resumen)
    workflow.add_node("conversacional", bound_nodo_conversacional)
    workflow.add_node("correo", bound_nodo_correo)

    # Definir el punto de entrada
    workflow.set_entry_point("clasificador")

    # Añadir las aristas condicionales (las decisiones)
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

    # Añadir las aristas normales (flujos directos)
    workflow.add_edge("analista", "validador") # Después del análisis, siempre se valida
    workflow.add_edge("validador", END)
    workflow.add_edge("resumen_tabla", END)
    workflow.add_edge("conversacional", END)
    workflow.add_edge("correo", END)

    # Compilar el grafo en una aplicación ejecutable
    return workflow.compile()