"""
================================================================================
 components/single_view.py — TAB 1: Análisis de comentario único
================================================================================
 Permite al usuario pegar UNA reseña y ver:
   - Sentimiento detectado (con color y emoji semántico)
   - Confianza del modelo en %
   - Idioma detectado (con badge "traducido al inglés" si aplica)
   - Probabilidades por clase (gráfica de barras horizontales)
   - El texto exacto que se le pasó al modelo (debugging / transparencia)

 Pensado para casos como:
   - Probar el modelo rápidamente con una reseña suelta
   - Integraciones puntuales (un POS que envía una reseña a la vez)
   - Demostraciones interactivas
================================================================================
"""

import streamlit as st                              # Framework de UI
from utils.viz import (                             # Funciones de gráficas reutilizables
    grafico_probabilidades_single,
    COLOR_POSITIVO,
    COLOR_NEUTRAL,
    COLOR_NEGATIVO,
    MAPA_COLORES,
)


# ============================================================================
# RESEÑAS DE EJEMPLO — para que el usuario pruebe sin pensar
# ============================================================================
# Una buena UX deja al usuario probar la app con un click sin tener que escribir.
# Estas reseñas cubren casos típicos: positiva clara, negativa fuerte, neutral ambigua,
# y multilingüe (para demostrar la capacidad ES→EN).

RESEÑAS_EJEMPLO = {
    "✨ Positiva (ES)": (
        "¡La mejor hamburguesa que he probado en años! El servicio fue rápido y "
        "el personal súper amable. Las papas fritas estaban perfectas. ¡Volveré pronto!"
    ),
    "⚠️ Negativa (EN)": (
        "The chicken sandwich was cold when it arrived and my order was completely wrong. "
        "Waited 25 minutes for a simple combo. Very disappointing experience overall."
    ),
    "🤔 Neutral (ES)": (
        "Comida normal, nada del otro mundo pero tampoco mala. El precio es aceptable "
        "y el servicio fue ni bueno ni malo. Probablemente no vuelva pronto."
    ),
    "🌐 Mixta (EN)": (
        "The food was great but the service was terrible. Waited forever to be seated "
        "but once we got our meal, it was actually delicious. Mixed feelings."
    ),
}


# ============================================================================
# FUNCIÓN PRINCIPAL DE LA TAB
# ============================================================================

def render(pipeline):
    """
    Dibuja toda la UI de la Tab 1 dentro del contenedor de Streamlit activo.

    Args:
        pipeline: instancia de PipelineSentimientos cargada en app.py
                  (se inyecta para evitar recargarla en cada render).
    """

    # --- Encabezado de la tab ---
    # st.markdown permite HTML con `unsafe_allow_html=True`. Lo usamos solo para
    # el título estilizado; el resto del contenido es texto plano por seguridad.
    st.markdown("### Análisis de comentario único")
    st.caption(
        "Pega cualquier reseña en español o inglés. El modelo detecta el idioma, "
        "la traduce si es necesario, y predice el sentimiento."
    )

    # --- Selector de ejemplos pre-cargados ---
    # st.columns crea layouts horizontales. Aquí hacemos 2 columnas:
    # - col1 (más ancha) para el text_area
    # - col2 para el selector de ejemplos
    # La proporción [3, 1] significa col1 ocupa 3/4 del ancho, col2 ocupa 1/4.
    col_input, col_ejemplos = st.columns([3, 1])

    with col_ejemplos:
        st.write("**Ejemplos:**")
        # Iteramos sobre los ejemplos pre-cargados creando un botón por cada uno.
        # Al hacer click, guardamos el texto en session_state para que el text_area lo lea.
        for nombre, texto in RESEÑAS_EJEMPLO.items():
            # `use_container_width=True` hace que el botón ocupe el ancho de la columna,
            # quedando todos alineados verticalmente (más ordenado).
            # `key` debe ser único por widget para que Streamlit lo identifique entre reruns.
            if st.button(nombre, key=f"ejemplo_{nombre}", use_container_width=True):
                # Guardamos el texto seleccionado en session_state.
                # Streamlit re-ejecuta el script desde arriba en cada interacción,
                # así que session_state es la única forma de PERSISTIR estado entre reruns.
                st.session_state.texto_single = texto

    with col_input:
        # text_area con valor por defecto desde session_state (vacío si no hay).
        # `value=...` establece el valor inicial; el usuario puede editarlo.
        # `height=200` da espacio para reseñas largas sin sobredimensionar la página.
        # `key='texto_single'` vincula este widget a session_state.texto_single,
        # permitiendo que los botones de ejemplo lo modifiquen.
        texto = st.text_area(
            "Reseña a analizar:",
            value=st.session_state.get('texto_single', ''),
            height=200,
            key='texto_single',
            placeholder="Ejemplo: La hamburguesa estaba deliciosa y el servicio fue excelente...",
        )

    # --- Botón de análisis ---
    # `type="primary"` usa el color de acento de Streamlit (azul por default).
    # Lo destacamos así porque es la acción principal de la tab.
    boton_analizar = st.button("Analizar sentimiento", type="primary", use_container_width=False)

    # --- Lógica de análisis ---
    # Solo procesamos si el usuario tocó el botón Y hay texto.
    if boton_analizar:
        # Validación: si el texto está vacío o es muy corto, mostramos warning sin llamar al modelo.
        # .strip() quita espacios iniciales/finales para no contar como contenido válido.
        if not texto or len(texto.strip()) < 10:
            st.warning("⚠️ El texto debe tener al menos 10 caracteres para ser analizado.")
            return  # salimos de la función, no hay nada más que mostrar

        # st.spinner muestra un loading animado mientras corre el bloque interno.
        # Es CRÍTICO en UX porque la inferencia puede tardar 1-3 segundos en CPU.
        with st.spinner("Procesando..."):
            resultado = pipeline.predecir(texto)

        # Si el pipeline retornó un error (ej: texto inválido tras limpieza), lo mostramos.
        if 'error' in resultado:
            st.error(f"❌ {resultado['error']}")
            return

        # --- Mostrar resultado en formato visual ---
        _mostrar_resultado(resultado)


# ============================================================================
# RENDERIZADO DEL RESULTADO
# ============================================================================

def _mostrar_resultado(resultado: dict):
    """
    Renderiza el resultado de una predicción de forma visualmente atractiva.

    Función privada (prefijo `_`) porque solo se usa dentro de este módulo.
    La separamos del flujo principal para mantener `render()` legible.

    Args:
        resultado: dict que retorna pipeline.predecir().
    """
    # Línea divisoria visual entre input y output.
    st.divider()

    # Extraemos los campos clave para no escribir resultado['...'] todo el tiempo.
    sentimiento = resultado['sentimiento']
    confianza = resultado['confianza']
    idioma = resultado['idioma_detectado']
    fue_traducido = resultado['traducido']

    # Mapeo de sentimiento → emoji + label en español.
    # Estos emojis cumplen una función UX: el usuario los reconoce ANTES que el texto.
    emoji_sent = {
        'positivo': '✅',
        'neutral': '⚖️',
        'negativo': '❌',
    }.get(sentimiento, '❓')

    # ---- Layout de 3 columnas para los KPIs principales del resultado ----
    # st.metric es un widget Streamlit que muestra un valor numérico grande con label.
    # Es el componente PERFECTO para mostrar 1 KPI clave de forma destacada.
    col1, col2, col3 = st.columns(3)

    with col1:
        # markdown con HTML inline para personalizar el color del sentimiento.
        # Esto requiere unsafe_allow_html=True. En general lo evitamos por seguridad,
        # pero acá el contenido es 100% generado por nosotros (no input del usuario).
        color = MAPA_COLORES.get(sentimiento, '#000')
        st.markdown(
            f"""
            <div style='padding: 12px; border-radius: 8px; background: rgba(0,0,0,0.02);
                        border-left: 4px solid {color};'>
                <p style='margin:0; font-size:13px; color:#666;'>Sentimiento</p>
                <p style='margin:0; font-size:28px; font-weight:600; color:{color};'>
                    {emoji_sent} {sentimiento.upper()}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Confianza como número grande con porcentaje. st.metric ya viene con buen estilo.
        # `delta` es el cambio respecto a un valor anterior (no aplica acá → omitimos).
        st.metric(
            label="Confianza del modelo",
            value=f"{confianza*100:.1f}%",
            help="Probabilidad asignada por el modelo a la clase ganadora. "
                 "Valores < 75% suelen indicar reseñas ambiguas.",
        )

    with col3:
        # Idioma detectado + indicador de traducción.
        # Usamos un mapeo simple código ISO → nombre legible.
        idiomas_map = {'en': 'Inglés', 'es': 'Español', 'fr': 'Francés', 'pt': 'Portugués'}
        idioma_nombre = idiomas_map.get(idioma, idioma.upper() if idioma else 'Desconocido')
        traduccion_str = " (traducido al inglés)" if fue_traducido else ""
        st.metric(
            label="Idioma detectado",
            value=idioma_nombre,
            delta=traduccion_str if fue_traducido else None,
            delta_color="off",  # 'off' = gris neutro (no verde/rojo)
        )

    # ---- Gráfica de probabilidades ----
    # Spacer para separar los KPIs de la gráfica.
    st.markdown("##### Probabilidades por clase")

    # Generamos la gráfica con la función centralizada de viz.py.
    # st.plotly_chart la renderiza. `use_container_width=True` la hace responsive.
    figura = grafico_probabilidades_single(resultado['probabilidades'])
    st.plotly_chart(figura, use_container_width=True)

    # ---- Detalle expandible (debugging / transparencia) ----
    # st.expander crea una sección colapsada por default. Útil para info "secundaria"
    # que no debe distraer pero está disponible para usuarios avanzados.
    with st.expander("🔧 Ver detalles del procesamiento"):
        # Mostramos qué texto exactamente vio el modelo, después de limpieza/traducción.
        # Esto ayuda a depurar casos donde la predicción parece extraña.
        st.markdown("**Texto original:**")
        st.code(resultado['texto_original'], language=None)

        st.markdown("**Texto después de limpieza:**")
        st.code(resultado['texto_limpio'], language=None)

        # Solo mostramos el "texto modelo" si fue diferente al limpio (o sea, hubo traducción).
        if fue_traducido:
            st.markdown("**Texto traducido al inglés (lo que ve el modelo):**")
            st.code(resultado['texto_modelo'], language=None)

        # Probabilidades exactas (4 decimales) para análisis técnico.
        st.markdown("**Probabilidades exactas:**")
        # st.json renderiza dicts con sintaxis highlighting y colapso de niveles.
        st.json(resultado['probabilidades'])
