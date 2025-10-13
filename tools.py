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

import streamlit as st # <-- Importante: Añadido para leer los secretos
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
    # ... (Tu código de clasificación original - sin cambios)
    prompt_orq = f"""
Clasifica la intención del usuario en UNA SOLA PALABRA...
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
    # Asegúrate de pegar aquí tu prompt de SQL completo con todas las reglas de negocio
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) sobre la tabla `automundial` para responder la pregunta del usuario.
    
    <<< REGLAS CRÍTICAS, DE FECHA, DE PRODUCTO, ETC. >>>
    (Pega aquí el prompt largo y detallado que tenías en tu archivo original)

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
    # ... (Tu código de análisis original - sin cambios)
    print("Generando análisis de datos...")
    preview = _df_preview(df, 50) or "(sin datos para analizar)"
    prompt_analisis = f"""
    Eres IANA, un analista de datos senior EXTREMADAMENTE PRECISO.
    (Pega aquí tus reglas de precisión y formato obligatorio)
    Pregunta Original: {pregunta_usuario}
    {hist_text}
    Datos para tu análisis (usa SÓLO estos):
    {preview}
    """
    try:
        analisis = llm_analista.invoke(prompt_analisis).content
        return analisis
    except Exception as e:
        print(f"Error en el análisis: {e}")
        return f"No pude generar el análisis. Error: {e}"


def generar_resumen_tabla(pregunta_usuario: str, res: dict, llm_analista: ChatOpenAI) -> dict:
    # ... (Tu código de resumen original - sin cambios)
    print("Generando resumen de tabla...")
    df = res.get("df")
    if df is None or df.empty:
        return res
    prompt = f"""
    Actúa como IANA, un analista de datos amable.
    (Pega aquí tus ejemplos de respuestas variadas)
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
    # ... (Tu código de conversación original - sin cambios)
    print("Generando respuesta conversacional...")
    prompt_personalidad = f"""Tu nombre es IANA, una IA amable de automundial...
    {hist_text}
    Pregunta: "{pregunta_usuario}" """
    try:
        respuesta = llm_analista.invoke(prompt_personalidad).content
        return {"texto": respuesta}
    except Exception as e:
        print(f"Error en conversación: {e}")
        return {"texto": f"Lo siento, hubo un problema. Error: {e}"}

# --- FUNCIONES DE CORREO (LÓGICA RESTAURADA) ---

def extraer_detalles_correo(pregunta_usuario: str, llm_analista: ChatOpenAI) -> dict:
    print("Extrayendo detalles para el correo...")
    contactos = dict(st.secrets.get("named_recipients", {}))
    default_recipient_name = st.secrets.get("email_credentials", {}).get("default_recipient", "")
    
    prompt = f"""
    Tu tarea es analizar la pregunta de un usuario y extraer los detalles para enviar un correo...
    Agenda de Contactos Disponibles: {', '.join(contactos.keys())}
    Pregunta del usuario: "{pregunta_usuario}"
    (Pega aquí el resto de tu prompt de extracción de detalles)
    """
    try:
        response = llm_analista.invoke(prompt).content
        json_response = response.strip().replace("```json", "").replace("```", "").strip()
        details = json.loads(json_response)
        
        recipient_identifier = details.get("recipient_name", "default")
        
        if "@" in recipient_identifier:
            final_recipient = recipient_identifier
        elif recipient_identifier in contactos:
            final_recipient = contactos[recipient_identifier]
        else:
            final_recipient = default_recipient_name

        return {
            "recipient": final_recipient,
            "subject": details.get("subject", "Reporte de Datos - IANA"),
            "body": details.get("body", "Adjunto encontrarás los datos solicitados.")
        }
    except Exception as e:
        print(f"Error extrayendo detalles del correo: {e}")
        return {
            "recipient": default_recipient_name,
            "subject": "Reporte de Datos - IANA",
            "body": "Adjunto encontrarás los datos solicitados."
        }

def enviar_correo_agente(recipient: str, subject: str, body: str, df: Optional[pd.DataFrame] = None) -> dict:
    print(f"Enviando correo a {recipient}...")
    try:
        creds = st.secrets["email_credentials"]
        sender_email = creds["sender_email"]
        sender_password = creds["sender_password"]
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        if df is not None and not df.empty:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            attachment = MIMEApplication(csv_buffer.getvalue(), _subtype='csv')
            attachment.add_header('Content-Disposition', 'attachment', filename="datos_iana.csv")
            msg.attach(attachment)
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return {"texto": f"¡Listo! El correo fue enviado exitosamente a {recipient}."}
    except Exception as e:
        print(f"Error al enviar correo: {e}")
        return {"texto": f"Lo siento, no pude enviar el correo. Detalle del error: {e}"}

# Nota: La función validar_y_corregir_respuesta_analista no se está usando en el grafo actual,
# pero la puedes dejar aquí para usarla en el futuro.
def validar_y_corregir_respuesta_analista(pregunta_usuario: str, res_analisis: dict, hist_text: str):
    # ... Tu lógica de validación aquí ...
    pass
