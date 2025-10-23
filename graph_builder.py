import pandas as pd
from typing import TypedDict, Optional, List
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
    validar_y_corregir_respuesta_analista # ¡Esta función ahora es crucial!
)

# ===============================================================
# 1. DEFINICIÓN DEL ESTADO DEL GRAFO (LA MEMORIA COMPARTIDA)
# ===============================================================
class AgentState(TypedDict):
    """
    Define la estructura de la memoria del grafo.
    (ACTUALIZADO con campos para el bucle de validación)
    """
    pregunta_usuario: str
    historial_chat: list
    clasificacion: str
    df: Optional[pd.DataFrame]
    analisis: Optional[str]
    respuesta_final: Optional[str]
    error: Optional[str]
    
    # --- Campos nuevos para el bucle de validación ---
    critica_validador: Optional[str]  # Guarda la crítica del supervisor
    decision_validador: Optional[str] # "aprobar" o "revisar"

# ===============================================================
# 2. DEFINICIÓN DE LOS NODOS DEL GRAFO (LOS AGENTES TRABAJADORES)
# ===============================================================

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
    
    res_datos = ejecutar_sql_real(pregunta, hist_text, llm_sql, db)
    
    if res_datos.get("df") is not None and not res_datos["df"].empty:
        return {"df": res_datos["df"]}
    else:
        return {"error": "No pude obtener datos para tu pregunta. Intenta reformularla."}

def nodo_analista(state: AgentState, llm_analista: ChatOpenAI):
    """
    (ACTUALIZADO) Genera un análisis experto. 
    Si recibe una crítica del validador, la usa para corregir su trabajo.
    """
    print("--- 📊 NODO: Analista de Datos ---")
    pregunta = state["pregunta_usuario"]
    hist_text = get_history_text(state["historial_chat"])
    df = state["df"]
    critica = state.get("critica_validador") # Obtiene la crítica (si existe)

    if critica:
        print(f"--- 📊 Analista (Corrigiendo): {critica} ---")
    
    # Llama a la herramienta (debe ser actualizada para aceptar 'critica')
    analisis_texto = analizar_con_datos(
        pregunta=pregunta, 
        hist_text=hist_text, 
        df=df, 
        llm=llm_analista, 
        critica=critica  # Pasa la crítica a la herramienta
    )
    
    # Limpia la crítica después de usarla
    return {"analisis": analisis_texto, "critica_validador": None}

def nodo_validador(state: AgentState, llm_validador: ChatOpenAI):
    """
    (ACTUALIZADO) Ya no es un paso directo.
    Llama a una herramienta de IA para validar la calidad del análisis.
    Decide si aprobarlo o enviarlo a revisión.
    """
    print("--- 🕵️‍♀️ NODO: Supervisor de Calidad / Validador ---")
    pregunta = state["pregunta_usuario"]
    analisis = state["analisis"]
    df = state["df"]
    
    # Llama a la herramienta de validación
    resultado_validacion = validar_y_corregir_respuesta_analista(
        llm=llm_validador,
        pregunta=pregunta,
        df=df,
        analisis=analisis
    )
    
    # Se espera que la herramienta devuelva un dict: 
    # {"decision": "aprobar" | "revisar", "comentario": "..."}
    
    if resultado_validacion.get("decision") == "aprobar":
        print("--- 🕵️‍♀️ Decisión: APROBADO ---")
        return {
            "respuesta_final": analisis, # Usa el análisis original aprobado
            "decision_validador": "aprobar"
        }
    else:
        print(f"--- 🕵️‍♀️ Decisión: REVISAR ({resultado_validacion.get('comentario')}) ---")
        return {
            "critica_validador": resultado_validacion.get("comentario"),
            "decision_validador": "revisar"
        }

def nodo_resumen_tabla(state: AgentState, llm_analista: ChatOpenAI):
    """Genera una introducción amable y conversacional para una tabla de datos."""
    print("--- ✍️ NODO: Resumen de Tabla ---")
    pregunta = state["pregunta_usuario"]
    res_datos = {"df": state["df"]}
    
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
    """Lee la clasificación del primer nodo y decide la ruta principal."""
    print(f"--- 🧭 DECISIÓN: Ruta basada en clasificación '{state['clasificacion']}' ---")
    if state.get("error"):
        return "end"
        
    clasificacion = state["clasificacion"]
    if clasificacion == "conversacional":
        return "conversacional"
    elif clasificacion == "correo":
        return "correo"
    elif clasificacion in ["analista", "consulta"]:
        return "consulta_sql"
    else:
        return "end"

def decidir_despues_sql(state: AgentState):
    """Una vez obtenidos los datos, decide si se necesita un análisis o solo un resumen."""
    print("--- 🧭 DECISIÓN: Ruta después de la consulta SQL ---")
    if state.get("error"):
        return "end"
        
    if state["clasificacion"] == "analista":
        return "analista"
    else: 
        return "resumen_tabla"

def decidir_despues_validacion(state: AgentState):
    """
    (NUEVA ARISTA) Decide si el flujo termina o vuelve al analista.
    """
    print("--- 🧭 DECISIÓN: Ruta después de la Validación ---")
    if state.get("decision_validador") == "aprobar":
        return "end"
    else:
        # ¡El bucle! Vuelve al analista para que corrija.
        return "analista"

# ===============================================================
# 4. CONSTRUCCIÓN Y COMPILACIÓN DEL GRAFO (ACTUALIZADO)
# ===============================================================
def create_graph(llms: dict, db: SQLDatabase):
    """
    Construye el diagrama de flujo ejecutable con el bucle de validación.
    """
    
    # "Bind" de los nodos con sus herramientas
    bound_nodo_clasificador = lambda state: nodo_clasificador(state, llms["orq"])
    bound_nodo_sql = lambda state: nodo_consulta_sql(state, llms["sql"], db)
    bound_nodo_analista = lambda state: nodo_analista(state, llms["analista"])
    # El validador ahora también necesita un LLM
    bound_nodo_validador = lambda state: nodo_validador(state, llms["validador"])
    bound_nodo_resumen = lambda state: nodo_resumen_tabla(state, llms["analista"])
    bound_nodo_conversacional = lambda state: nodo_conversacional(state, llms["analista"])
    bound_nodo_correo = lambda state: nodo_correo(state, llms["analista"])
    
    workflow = StateGraph(AgentState)

    # Añadir todos los nodos
    workflow.add_node("clasificador", bound_nodo_clasificador)
    workflow.add_node("consulta_sql", bound_nodo_sql)
    workflow.add_node("analista", bound_nodo_analista)
    workflow.add_node("validador", bound_nodo_validador) # Nodo actualizado
    workflow.add_node("resumen_tabla", bound_nodo_resumen)
    workflow.add_node("conversacional", bound_nodo_conversacional)
    workflow.add_node("correo", bound_nodo_correo)

    # Definir por dónde empieza siempre el flujo
    workflow.set_entry_point("clasificador")

    # --- Añadir Aristas Condicionales ---
    
    # 1. Decisión inicial (Clasificador)
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
    
    # 2. Decisión después del SQL
    workflow.add_conditional_edges(
        "consulta_sql",
        decidir_despues_sql,
        {
            "analista": "analista",
            "resumen_tabla": "resumen_tabla",
            "end": END
        }
    )
    
    # 3. (NUEVO) Decisión después del Validador (¡Aquí se crea el bucle!)
    workflow.add_conditional_edges(
        "validador",
        decidir_despues_validacion,
        {
            "analista": "analista", # Vuelve al analista
            "end": END              # Termina el flujo
        }
    )

    # --- Añadir Aristas Normales (Flechas directas) ---
    workflow.add_edge("analista", "validador") # El analista SIEMPRE va al validador
    
    # Nodos que terminan el flujo directamente
    workflow.add_edge("resumen_tabla", END)
    workflow.add_edge("conversacional", END)
    workflow.add_edge("correo", END)
    
    # (Se elimina la arista directa 'validador' -> END, ahora es condicional)

    # Compilar el grafo
    app = workflow.compile()

    return app

# ===============================================================
# 5. EXPORTAR EL GRAFO EN FORMATO MERMAID (VISUALIZACIÓN)
# ===============================================================
def export_graph_mermaid(graph):
    """
    Exporta el grafo de agentes a formato Mermaid para visualizarlo
    en Streamlit o en https://mermaid.live
    """
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        print("Código Mermaid del flujo:")
        print(mermaid_code)
        return mermaid_code
    except Exception as e:
        print(f"Error al exportar gráfico Mermaid: {e}")
        return None
