import pandas as pd
import re
import io
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Optional

import streamlit as st
from elevenlabs.client import ElevenLabs
from sqlalchemy import text
from langchain_community.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

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

def _df_preview(df: pd.DataFrame, n: int = 50) -> str:
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

        if any(pal in pregunta.lower() for pal in ["analiza", "compara", "resume", "explica", "por qué", "tendencia", "insights"]):
             return "analista"

        if r in opciones:
            return r
        
        if any(pal in pregunta.lower() for pal in ["venta", "ventas", "margen", "costo", "costos", "precio", "unidades", "rubro", "cliente", "artículo", "producto", "línea", "familia", "total", "facturado"]):
            return "consulta"
        
        return "conversacional" # Default más seguro
    except Exception:
        return "conversacional"


def ejecutar_sql_real(pregunta_usuario: str, hist_text: str, llm_sql: ChatOpenAI, db: SQLDatabase):
    print("Traduciendo pregunta a SQL...")
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) sobre la tabla `autollantas` para responder la pregunta del usuario.

    ---
    <<< NUEVA REGLA: PARA VALORES MONETARIOS >>>
     1. Cuando el usuario mencione “margen”, “margen bruto” o “ganancia bruta”, se debe consultar la información en la columna 'Porcentaje_Margen_Bruto', que representa el **margen relativo** (porcentaje de utilidad sobre ventas).  
        Si el usuario pide explícitamente “margen en pesos”, “margen monetario” o “margen absoluto”, entonces usa la columna 'Margen_Bruto', que representa el **margen absoluto** (valor monetario de la utilidad bruta).  
        Ejemplo:  
        - “Dame el margen bruto por mes” → usa `Porcentaje_Margen_Bruto`  
        - “Dame el margen absoluto en pesos” → usa `Margen_Bruto`
        
        ❗Nunca promedies el margen ni uses AVG(Porcentaje_Margen_Bruto).  
        El margen relativo SIEMPRE debe calcularse dinámicamente como:
         (1 - SUM(Costo_Reales) / SUM(Ventas_Reales)) * 100
        o equivalente:
         (SUM(Ventas_Reales - Costo_Reales) / SUM(Ventas_Reales)) * 100
        según el nivel de agrupación.
        Ejemplo:
        SELECT MONTH(Fecha) AS Mes,  
               (1 - SUM(Costo_Reales) / SUM(Ventas_Reales)) * 100 AS Margen_Porcentual
        FROM autollantas
        GROUP BY MONTH(Fecha);
        
     2. Cuando el usuario mencione “porcentaje de margen”, “% margen”, “margen porcentual” o “margen en porcentaje”, se debe consultar la información en la columna 'Porcentaje_Margen_Bruto', que representa la proporción del margen bruto sobre las ventas reales.
     3. Cuando el usuario mencione “unidades vendidas”, “cantidad de productos vendidos” o “número de ventas”, se está refiriendo al campo 'Unidades_Vendidas'.
     4. Cuando el usuario pregunte por “precio promedio”, “valor medio de venta” o “promedio de precios”, se refiere al campo 'Precio_Promedio', que corresponde al promedio del valor unitario de las ventas.
     5. Cuando el usuario mencione “ventas reales”, “ventas totales” o “valor vendido”, se está refiriendo al campo 'Ventas_Reales', que representa el total monetario facturado o reconocido como ingreso real.
     6. Cuando el usuario mencione “costos reales”, “costos totales” o “valor del costo”, se refiere al campo Costo_Reales, que muestra el total de costos asociados a las ventas (sin incluir margen ni impuestos).
     7. Ejemplo: Si la pregunta es "¿cuál es el total facturado?", la consulta debería ser algo como `SELECT SUM(Ventas_Reales) FROM autollantas;`. Aplica este patrón a otras métricas.
    ---
    <<< REGLA CRÍTICA PARA FILTRAR POR FECHA >>>
    1. Tu tabla tiene una columna de fecha llamada `Fecha`.
    2. Si el usuario especifica un año (ej: "del 2025", "en 2024"), SIEMPRE debes añadir una condición `WHERE YEAR(Fecha) = [año]` a la consulta.
    3. Ejemplo: "dame las ventas de 2025" -> DEBE INCLUIR `WHERE YEAR(Fecha) = 2025`.
    ---
    <<< REGLA DE ORO PARA BÚSQUEDA DE PRODUCTOS >>>
    1. Cuando el usuario mencione “artículo”, “producto”, “ítem”, “referencia”, “nombre del repuesto” o “nombre del material”, se está refiriendo al campo 'Nombre_Articulo', el cual contiene el nombre comercial o técnico de cada producto registrado en inventario o en las órdenes.
       Este campo puede incluir detalles como:
       - Medidas o especificaciones (ej. 195/60R16, 11R-22.5)
       - Marca o fabricante (ej. Yokohama, Firestone, Alliance)
       - Tipo o aplicación (ej. filtro de combustible, llanta, aire, repuesto)
    2. Si el usuario pregunta por un producto específico, usa `WHERE LOWER(Nombre_Articulo) LIKE '%palabra%'.
    3. Cuando el usuario mencione “cliente”, “empresa”, “razón social”, “comprador”, “contratante” o “nombre del cliente”, se está refiriendo al campo 'Nombre_Cliente', que representa la entidad (persona natural o jurídica) a la que se le vendió, facturó o prestó un servicio.
    4. Cuando el usuario mencione “línea”, “marca”, “familia de producto”, “referencia comercial” o “proveedor principal”, se está refiriendo al campo 'Nombre_Linea', el cual identifica la marca, línea o categoría principal a la que pertenece un artículo.
    ---
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

# ==========================================================
# FUNCIÓN ACTUALIZADA (acepta 'critica')
# ==========================================================
def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: Optional[pd.DataFrame], llm_analista: ChatOpenAI, critica: Optional[str] = None) -> str:
    print("Generando análisis de datos...")
    preview = _df_preview(df) or "(sin datos para analizar)"
    
    # Construcción modular del prompt
    prompt_base = f"""
    Eres IANA, un analista de datos senior EXTREMADAMENTE PRECISO y riguroso.
    ---
    <<< REGLAS CRÍTICAS DE PRECISIÓN >>>
    1. **NO ALUCINAR**: NUNCA inventes números, totales, porcentajes o nombres de productos/categorías que no estén EXPRESAMENTE en la tabla de 'Datos'.
    2. **DATOS INCOMPLETOS**: Reporta los vacíos (p.ej., "sin datos para Marzo") sin inventar valores.
    3. **VERIFICAR CÁLCULOS**: Antes de escribir un número, revisa el cálculo (sumas/conteos/promedios) con los datos.
    4. **CITAR DATOS**: Basa CADA afirmación que hagas en los datos visibles en la tabla.
    ---
    Pregunta Original: {pregunta_usuario}
    {hist_text}
    Datos para tu análisis (usa SÓLO estos):
    {preview}
    ---
    """

    prompt_formato = """
    FORMATO OBLIGATORIO:
    Entregar el resultado en 3 bloques:
    📌 **Resumen Ejecutivo**: El hallazgo principal (el "bottom line") en una o dos frases, con los números más importantes.
    🔍 **Números Clave**: Una lista corta (bullet points) de los totales, promedios, o ratios más relevantes que soportan tu resumen.
    ⚠ **Observaciones Importantes** (Opcional): Si notas algo atípico, una concentración alta (ej. 80% en un cliente) o un dato faltante, menciónalo aquí.

    Sé muy breve, directo y diciente.
    """

    # Si hay una crítica del supervisor, se añade al prompt.
    if critica:
        print(f"--- Analista Recibiendo Crítica: {critica} ---")
        prompt_correccion = f"""
        <<< ATENCIÓN: CORRECCIÓN REQUERIDA >>>
        Tu análisis anterior fue rechazado por el supervisor de calidad.
        Crítica del supervisor: "{critica}"
        
        Por favor, genera un NUEVO análisis corrigiendo este punto. Sé muy riguroso con la crítica y revisa TODOS los números de nuevo.
        <<< FIN DE LA CORRECCIÓN >>>
        """
        prompt_analisis = prompt_base + prompt_correccion + prompt_formato
    else:
        prompt_analisis = prompt_base + prompt_formato

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
    <<< REGLA CRÍTICA >>>
    Tu ÚNICA tarea es escribir una frase CORTA y amable que sirva como introducción a la tabla de datos que se mostrará.
    NO respondas la pregunta original del usuario de forma general. Sé breve, directo y conversacional. Tu respuesta debe ser una sola línea.

    Pregunta original del usuario: "{pregunta_usuario}"
    
    ---
    Aquí tienes varios ejemplos de cómo responder:

    Ejemplo 1:
    Pregunta: "cuáles son los proveedores"
    Respuesta: "¡Listo! Aquí tienes la lista de proveedores que encontré:"

    Ejemplo 2:
    Pregunta: "y sus ventas?"
    Respuesta: "He consultado las cifras de ventas. Te las muestro en la siguiente tabla:"
    
    Ejemplo 3:
    Pregunta: "dame el total por mes"
    Respuesta: "Claro que sí. He preparado una tabla con los totales por mes:"
    ---

    Ahora, genera la introducción para la pregunta del usuario actual. Sé breve.
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
    prompt_personalidad = f"""Tu nombre es IANA, una IA amable de autollantas. Ayuda a analizar datos.
    Si el usuario hace un comentario casual, responde amablemente de forma natural, muy humana y redirígelo a tus capacidades.
    {hist_text}
    Pregunta: "{pregunta_usuario}" """
    try:
        respuesta = llm_analista.invoke(prompt_personalidad).content
        return {"texto": respuesta}
    except Exception as e:
        print(f"Error en conversación: {e}")
        return {"texto": f"Lo siento, hubo un problema. Error: {e}"}


def extraer_detalles_correo(pregunta_usuario: str, llm_analista: ChatOpenAI) -> dict:
    print("Extrayendo detalles para el correo...")
    contactos = dict(st.secrets.get("named_recipients", {}))
    default_recipient = st.secrets.get("email_credentials", {}).get("default_recipient", "")
    
    prompt = f"""
    Tu tarea es analizar la pregunta de un usuario y extraer los detalles para enviar un correo. Tu output DEBE SER un JSON válido.

    Agenda de Contactos Disponibles: {', '.join(contactos.keys())}

    Pregunta del usuario: "{pregunta_usuario}"

    Instrucciones para extraer:
    1.  `recipient_name`: Busca un nombre de la "Agenda de Contactos" en la pregunta. Si encuentras un nombre (ej: "Oscar"), pon ese nombre aquí. Si encuentras una dirección de correo explícita (ej: "test@test.com"), pon la dirección completa aquí. Si no encuentras ni nombre ni correo, usa "default".
    2.  `subject`: Crea un asunto corto y descriptivo basado en la pregunta.
    3.  `body`: Crea un cuerpo de texto breve y profesional para el correo.

    Ejemplo:
    Pregunta: "envía el reporte a Oscar por favor"
    JSON Output:
    {{
        "recipient_name": "Oscar",
        "subject": "Reporte de Datos Solicitado",
        "body": "Hola, como solicitaste, aquí tienes el reporte con los datos."
    }}
    
    JSON Output para la pregunta actual:
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
            final_recipient = default_recipient

        return {
            "recipient": final_recipient,
            "subject": details.get("subject", "Reporte de Datos - IANA"),
            "body": details.get("body", "Adjunto encontrarás los datos solicitados.")
        }
    except Exception as e:
        print(f"Error extrayendo detalles del correo: {e}")
        return {
            "recipient": default_recipient,
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

# ==========================================================
# FUNCIÓN ACTUALIZADA (Implementación del Supervisor QA)
# ==========================================================
def validar_y_corregir_respuesta_analista(llm: ChatOpenAI, pregunta: str, df: Optional[pd.DataFrame], analisis: str) -> dict:
    """
    Actúa como un supervisor de QA. Valida la respuesta del analista contra los datos
    y decide si aprobarla o enviarla a revisión.
    """
    print("--- 🕵️‍♀️ Supervisor QA: Validando análisis... ---")
    preview = _df_preview(df) or "(sin datos)"
    
    prompt_validador = f"""
    Eres un Supervisor de Calidad de IA (QA) EXTREMADAMENTE estricto.
    Tu trabajo es validar si la respuesta de un "Agente Analista" es precisa, completa y se basa ESTRICTAMENTE en los datos proporcionados.

    ---
    1. Pregunta Original del Usuario:
    "{pregunta}"

    2. Datos que el analista usó (previsualización):
    "{preview}"

    3. Análisis generado por el Agente Analista (el que debes validar):
    "{analisis}"
    ---

    REGLAS DE VALIDACIÓN:
    1.  **Precisión Absoluta**: ¿Son correctos TODOS los números, totales, promedios y porcentajes mencionados en el análisis, basándose en los "Datos"?
    2.  **No Alucinación**: ¿El análisis menciona algún producto, cliente, mes o dato que NO esté en la tabla de "Datos"? Si lo hace, es un error grave.
    3.  **Respuesta Completa**: ¿El análisis responde directamente a la "Pregunta Original del Usuario"?
    4.  **Tono**: ¿El análisis es profesional y directo?

    ---
    TU TAREA:
    Responde con un JSON. Tu respuesta debe ser SOLO el JSON.

    Si el análisis es PERFECTO (cumple todas las reglas):
    {{
      "decision": "aprobar",
      "comentario": "El análisis es preciso y responde bien a la pregunta."
    }}

    Si el análisis tiene CUALQUIER error (alucinación, cálculo incorrecto, no responde la pregunta):
    {{
      "decision": "revisar",
      "comentario": "[Explica BREVEMENTE el error. Ej: 'El total de ventas es incorrecto, la suma da X y no Y.' o 'El análisis menciona a un cliente que no está en los datos.' o 'El análisis no responde laf pregunta sobre el margen.']"
    }}

    JSON de Decisión:
    """
    
    try:
        response_str = llm.invoke(prompt_validador).content
        json_response = response_str.strip().replace("```json", "").replace("```", "").strip()
        decision = json.loads(json_response)
        
        if decision.get("decision") not in ["aprobar", "revisar"]:
            print("Decisión inválida del validador. Aprobando por defecto.")
            decision = {"decision": "aprobar", "comentario": "Decisión de validador inválida."}
        
        return decision
    except Exception as e:
        print(f"Error en el validador, aprobando por defecto: {e}")
        # Fallback seguro: si el validador falla, aprueba la respuesta para no bloquear al usuario.
        return {"decision": "aprobar", "comentario": "Validación fallida, aprobado por defecto."}


def text_to_audio_elevenlabs(text: str) -> bytes:
    """
    Convierte un texto en audio usando la API de ElevenLabs.
    Devuelve los bytes del archivo de audio (ej. un MP3).
    """
    try:
        # Se configura el cliente con la API key de los secretos
        client = ElevenLabs(
            api_key=st.secrets["elevenlabs_api_key"]
        )

        # Se genera el audio a partir del texto
        audio_bytes = client.generate(
            text=text,
            voice="Rachel",  # Puedes usar cualquier voz. "Rachel" es una popular en inglés.
            model="eleven_multilingual_v2" # Un buen modelo para múltiples idiomas
        )
        return audio_bytes

    except Exception as e:
        print(f"Error al generar audio con ElevenLabs: {e}")
        return None







