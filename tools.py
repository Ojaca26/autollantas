# tools.py

import pandas as pd
import re
import io
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Optional

from sqlalchemy import text
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# ============================================
# FUNCIONES AUXILIARES PARA DATOS
# ============================================

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(r'[\u00A0\s]', '', regex=True).str.replace(',', '', regex=False).str.replace('$', '', regex=False).str.replace('%', '', regex=False)
    try:
        return pd.to_numeric(s2)
    except Exception:
        return s

def get_history_text(chat_history: list, n_turns=3) -> str:
    if not chat_history or len(chat_history) <= 1:
        return ""
    history_text = []
    relevant_history = chat_history[-(n_turns * 2 + 1) : -1]
    for msg in relevant_history:
        content = msg.get("content", {})
        text_content = ""
        if isinstance(content, dict):
            text_content = content.get("texto", "")
        elif isinstance(content, str):
            text_content = content
        if text_content:
            role = "Usuario" if msg["role"] == "user" else "IANA"
            history_text.append(f"{role}: {text_content}")
    if not history_text:
        return ""
    return "\n--- Contexto de Conversación Anterior ---\n" + "\n".join(history_text) + "\n--- Fin del Contexto ---\n"

def _df_preview(df: pd.DataFrame, n: int = 5) -> str:
    if df is None or df.empty:
        return ""
    try:
        return df.head(n).to_markdown(index=False)
    except Exception:
        return df.head(n).to_string(index=False)

def _asegurar_select_only(sql: str) -> str:
    sql_clean = sql.strip().rstrip(';')
    if not re.match(r'(?is)^\s*select\b', sql_clean):
        raise ValueError("Solo se permite ejecutar consultas SELECT.")
    sql_clean = re.sub(r'(?is)\blimit\s+\d+\s*$', '', sql_clean).strip()
    return sql_clean

# ============================================
# HERRAMIENTAS DE AGENTES (WORKERS)
# ============================================

def clasificar_intencion(pregunta: str, llm_orq: ChatOpenAI) -> str:
    prompt_orq = f"""
Clasifica la intención del usuario en UNA SOLA PALABRA. Presta especial atención a los verbos de acción y palabras clave.

1. `analista`: Si la pregunta pide explícitamente una interpretación, resumen, comparación o explicación.
   PALABRAS CLAVE PRIORITARIAS: analiza, compara, resume, explica, por qué, tendencia, insights, dame un análisis, haz un resumen, interpreta.
   Si una de estas palabras clave está presente, la intención SIEMPRE es `analista`.

2. `consulta`: Si la pregunta pide datos crudos (listas, conteos, totales, valores, métricas) o resultados numéricos directos, y NO contiene palabras clave de `analista`.
   Ejemplos: 'cuántos proveedores hay', 'lista todos los productos', 'muéstrame el total', 'ventas por mes', 'margen por cliente', 'costo total', 'precio promedio'.
   PALABRAS CLAVE ADICIONALES: venta, ventas, costo, costos, margen, precio, unidades, rubro, cliente, artículo, producto, línea, familia, total, facturado, utilidad.

3. `correo`: Si la pregunta pide explícitamente enviar un correo, email o reporte.
   PALABRAS CLAVE: envía, mandar, correo, email, reporte a, envíale a.

4. `conversacional`: Si es un saludo o una pregunta general no relacionada con datos.
   Ejemplos: 'hola', 'gracias', 'qué puedes hacer', 'cómo estás'.

Pregunta: "{pregunta}"
Clasificación:
"""
    try:
        opciones = {"consulta", "analista", "conversacional", "correo"}
        r = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")

        if any(pal in pregunta.lower() for pal in ["venta", "ventas", "margen", "costo", "costos", "precio", "unidades", "rubro", "cliente", "artículo", "producto", "línea", "familia", "total", "facturado"]):
            return "consulta"
            
        return r if r in opciones else "consulta"
    except Exception:
        return "consulta"


def ejecutar_sql_real(pregunta_usuario: str, hist_text: str, llm_sql: ChatOpenAI, db: SQLDatabase):
    print("Traduciendo pregunta a SQL...")
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) sobre la tabla `automundial` para responder la pregunta del usuario.
    [... Aquí van todas tus reglas de negocio para SQL (margen, fechas, productos, etc.) ...]
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"
    Devuelve SOLO la consulta SQL (sin explicaciones).
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query_bruta = query_chain.invoke({"question": prompt_con_instrucciones})
        m = re.search(r'(?is)(select\b.+)$', sql_query_bruta.strip())
        sql_query_limpia = m.group(1).strip() if m else sql_query_bruta.strip()
        sql_query_limpia = re.sub(r'(?is)^```sql|```$', '', sql_query_limpia).strip()
        sql_query_limpia = _asegurar_select_only(sql_query_limpia)
        
        print(f"Ejecutando SQL: {sql_query_limpia}")
        with db._engine.connect() as conn:
            df = pd.read_sql(text(sql_query_limpia), conn)
        
        # Lógica para añadir la fila de totales
        if not df.empty:
            value_cols = [c for c in df.select_dtypes("number").columns if not re.search(r"(?i)\b(mes|año|dia|fecha)\b", c)]
            if value_cols:
                total_row = {col: df[col].sum() if col in value_cols else "" for col in df.columns}
                total_row[df.columns[0]] = "Total"
                df.loc[len(df)] = total_row
        
        return {"sql": sql_query_limpia, "df": df}
    except Exception as e:
        print(f"Error al ejecutar SQL: {e}")
        return {"sql": None, "df": None, "error": str(e)}


def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: Optional[pd.DataFrame], llm_analista: ChatOpenAI) -> str:
    print("Generando análisis de datos...")
    preview = _df_preview(df, 50) or "(sin datos para analizar)"
    prompt_analisis = f"""
    Eres IANA, un analista de datos senior EXTREMADAMENTE PRECISO.
    [... Aquí van todas tus reglas críticas de precisión (NO ALUCINAR, etc.) ...]
    Pregunta Original: {pregunta_usuario}
    {hist_text}
    Datos para tu análisis (usa SÓLO estos):
    {preview}
    ---
    FORMATO OBLIGATORIO:
    [... Aquí va tu formato de resumen ejecutivo, números de referencia, etc. ...]
    """
    try:
        analisis = llm_analista.invoke(prompt_analisis).content
        return analisis
    except Exception as e:
        print(f"Error en el análisis: {e}")
        return f"No pude generar el análisis. Error: {e}"


def generar_resumen_tabla(pregunta_usuario: str, res: dict, llm_analista: ChatOpenAI) -> dict:
    print("Generando resumen de tabla...")
    df = res.get("df")
    if df is None or df.empty:
        return res
    prompt = f"""
    Actúa como IANA, un analista de datos amable. Escribe una breve introducción para la tabla.
    [... Aquí van todos tus ejemplos de respuestas variadas ...]
    Pregunta del usuario: "{pregunta_usuario}"
    Ahora, genera la introducción para la pregunta del usuario actual:
    """
    try:
        introduccion = llm_analista.invoke(prompt).content
        res["texto"] = introduccion
    except Exception as e:
        print(f"Error generando resumen: {e}")
        res["texto"] = "Aquí están los datos que solicitaste:"
    return res


def responder_conversacion(pregunta_usuario: str, hist_text: str, llm_analista: ChatOpenAI) -> dict:
    print("Generando respuesta conversacional...")
    prompt_personalidad = f"""Tu nombre es IANA, una IA amable de automundial.
    {hist_text}
    Pregunta: "{pregunta_usuario}" """
    try:
        respuesta = llm_analista.invoke(prompt_personalidad).content
        return {"texto": respuesta}
    except Exception as e:
        print(f"Error en conversación: {e}")
        return {"texto": f"Lo siento, hubo un problema. Error: {e}"}

# --- Funciones de Correo ---
# (Asumiendo que los secretos ahora se leen en `app.py` y se pasan si es necesario,
# o mejor aún, se leen desde el entorno directamente si es posible)

def extraer_detalles_correo(pregunta_usuario: str, llm_analista: ChatOpenAI) -> dict:
    # Esta función necesitaría acceso a los secretos para la agenda y el destinatario por defecto.
    # Por ahora, la dejamos así, pero en un despliegue real, se deben manejar los secretos de forma segura.
    print("Extrayendo detalles para el correo...")
    # Lógica simplificada
    return { "recipient": "destinatario@ejemplo.com", "subject": "Reporte de Datos", "body": "Adjunto los datos." }


def enviar_correo_agente(recipient: str, subject: str, body: str, df: Optional[pd.DataFrame] = None) -> dict:
    # Esta función también necesita secretos para las credenciales de envío.
    print(f"Enviando correo a {recipient}...")
    # Lógica simplificada para evitar manejar secretos aquí
    # En un caso real, aquí iría tu código de smtplib
    return {"texto": f"¡Listo! El correo fue enviado a {recipient} (simulación)."}