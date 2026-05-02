"""
================================================================================
 app.py — Entry point de la aplicación Streamlit
================================================================================
 Este es el archivo que se ejecuta con: `streamlit run app.py`

 Su responsabilidad es ORQUESTAR:
   1. Configurar la página (título, layout, ícono)
   2. Cargar el modelo UNA SOLA VEZ (cacheado con @st.cache_resource)
   3. Renderizar el sidebar (info del modelo, configuración global)
   4. Crear las 3 tabs y delegar el render a cada módulo

 La lógica de negocio está en los módulos `components/` y `utils/`.
 Este archivo es deliberadamente delgado para mantener la app modular.
================================================================================
"""

# --- Imports del framework ---
import streamlit as st                              # Framework de UI

# --- Imports del proyecto ---
# Importamos la clase del pipeline (definida en pipeline.py).
from pipeline import PipelineSentimientos, DISPOSITIVO

# Importamos los renderizadores de cada tab. Tener un módulo por tab mantiene
# `app.py` corto y permite trabajar en cada vista de forma independiente.
from components import single_view, batch_view, insights_view


# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
# st.set_page_config DEBE ser la primera llamada de Streamlit en el script,
# antes de cualquier otro st.*. Si no, Streamlit lanza una excepción.
#
# `layout='wide'` usa todo el ancho del navegador (vs el default 'centered' que
# limita a ~700px). Crítico para dashboards porque queremos espacio para gráficas.

st.set_page_config(
    page_title="Sentiment Pulse — Análisis de reseñas",
    page_icon="📊",                                  # ícono que aparece en la pestaña del navegador
    layout="wide",
    initial_sidebar_state="expanded",                # sidebar abierto al cargar (mejor descubribilidad)
    menu_items={
        # Personalizamos los items del menú (esquina sup-derecha).
        # Útil para enlazar a documentación o reportar bugs.
        'Get Help': 'https://github.com/tu-usuario/sentiment-pulse',
        'Report a bug': None,
        'About': (
            "**Sentiment Pulse** — TFM Big Data & Business Analytics, ADEN.\n\n"
            "Análisis de sentimientos para reseñas de restaurantes con DistilBERT."
        ),
    },
)


# ============================================================================
# CSS CUSTOM — pequeños ajustes de estilo
# ============================================================================
# Streamlit es muy bonito por default pero algunos detalles los queremos finetuned:
#   - Reducir padding superior (la app empieza muy abajo por default)
#   - Hacer las tabs un poco más prominentes
#
# Inyectamos CSS con st.markdown + unsafe_allow_html. Esto es seguro porque
# el CSS lo escribimos nosotros (no viene del usuario).

st.markdown(
    """
    <style>
        /* Reducir el padding superior para aprovechar mejor el espacio vertical */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Tabs: más espacio vertical y separación entre ellas */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            font-size: 15px;
        }
        /* Métricas: hacer el valor un poco más grande */
        [data-testid="stMetricValue"] {
            font-size: 28px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# CARGA DEL MODELO (cacheada)
# ============================================================================
# @st.cache_resource memoiza el objeto retornado durante TODO el ciclo de vida
# del proceso Streamlit (no solo durante la sesión del usuario).
#
# Diferencia clave vs @st.cache_data:
#   - @st.cache_data: para datos serializables (DataFrames, dicts, números). Hace COPIAS.
#   - @st.cache_resource: para objetos NO serializables o costosos (modelos ML, conexiones DB).
#                        NO hace copias — todos los usuarios comparten la misma instancia.
#
# Para modelos PyTorch usamos cache_resource porque:
#   1. Cargar DistilBERT (~265MB) toma 5-10s. No queremos hacerlo en cada interacción.
#   2. El objeto modelo no se serializa bien (es un grafo computacional con tensores en GPU).
#   3. Puede ser compartido entre usuarios sin problemas (es read-only durante inferencia).

@st.cache_resource(show_spinner="Cargando modelo de sentimientos...")
def cargar_pipeline(ruta_modelo: str) -> PipelineSentimientos:
    """
    Carga el pipeline una sola vez y lo cachea.

    Esta función se ejecuta UNA VEZ al iniciar la app (o cuando cambia ruta_modelo).
    En todas las invocaciones siguientes, Streamlit devuelve la misma instancia.

    Args:
        ruta_modelo: directorio donde está el modelo entrenado (config.json + pesos).

    Returns:
        Instancia de PipelineSentimientos lista para predecir.
    """
    return PipelineSentimientos(ruta_modelo)


# ============================================================================
# SIDEBAR — info del modelo y configuración global
# ============================================================================
# El sidebar es ideal para:
#   - Mostrar metadata persistente (qué modelo está cargado, en qué dispositivo)
#   - Configuración que NO cambia frecuentemente (ruta del modelo, tema, idioma)
#
# Lo separamos del contenido principal porque no queremos que distraiga del análisis.

with st.sidebar:
    # Logo / Título de la app.
    # Usamos markdown directo en lugar de st.title para tener control de tamaño.
    st.markdown("# 📊 Sentiment Pulse")
    st.caption("Análisis de reseñas de restaurantes")
    st.divider()

    # --- Configuración del modelo ---
    st.markdown("### ⚙️ Modelo")

    # Input para que el usuario cambie la ruta del modelo si tiene varios entrenamientos.
    # Default razonable: una carpeta hermana al app.py llamada 'modelo_sentimientos'.
    # Si tu modelo está en otra ruta (ej: Google Drive montado, S3, etc.), cámbialo aquí.
    #Esta es la ruta local que usé para desarrollo. En producción, el modelo se carga desde Hugging Face Hub (ver pipeline.py).
    # ruta_modelo = st.text_input(
    #     "Ruta al modelo entrenado:",
    #     value="./modelo_sentimientos",
    #     help="Carpeta que contiene config.json, model.safetensors, tokenizer.json, etc.",
    # )
    # ruta_modelo = st.text_input(
    #     "Ruta al modelo entrenado:",
    #     value= "sadojrtora/distilbert-sentimientosTFM",
    #     help="Carpeta que contiene config.json, model.safetensors, tokenizer.json, etc.",
    # )
    ruta_modelo = "sadojrtora/distilbert-sentimientosTFM"

    # Intentamos cargar el modelo. Si falla (carpeta no existe, archivos corruptos),
    # mostramos error claro y abortamos antes de renderizar las tabs.
    try:
        pipeline = cargar_pipeline(ruta_modelo)
        # Indicador visual de que el modelo está OK.
        st.success(f"✅ Modelo cargado")
        st.caption(f"📍 Dispositivo: `{DISPOSITIVO}`")  # cuda / cpu — útil para debugging
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo:\n\n{e}")
        st.info(
            "**Posibles causas:**\n"
            "- La ruta no existe o no contiene los archivos del modelo.\n"
            "- Falta alguna dependencia (`pip install -r requirements.txt`).\n"
            "- El modelo fue guardado con una versión incompatible de transformers."
        )
        # st.stop() detiene la ejecución acá. Las tabs no se renderizan.
        # Esto evita errores en cascada si las tabs intentan usar `pipeline` (que no existe).
        st.stop()

    st.divider()

    # --- Info del proyecto ---
    # Sección con info académica/contextual del TFM.
    # Útil cuando defiendes el trabajo o lo presentas a un cliente.
    st.markdown("### 📚 Acerca de")
    st.markdown(
        """
        **Modelo base:** DistilBERT (uncased)
        **Dataset:** Yelp Review Full (700k reseñas)
        **Clases:** Positivo · Neutral · Negativo
        **Idiomas:** Inglés (nativo) + Español (traducción ES→EN)
        """
    )

    st.caption(
        "TFM — Big Data & Business Analytics\n\n"
        "ADEN International Business School\n\n"
        "Donaldo Díaz · 2026"
    )


# ============================================================================
# CONTENIDO PRINCIPAL: 3 TABS
# ============================================================================
# Header de la app (visible en todas las tabs).
st.markdown("# 🍔 Sentiment Pulse")
st.caption(
    "Plataforma de análisis de sentimientos para reseñas de restaurantes. "
    "Sube reseñas, descubre patrones, toma decisiones."
)

# st.tabs crea un selector de pestañas. Recibe una lista de strings (nombres).
# Retorna una lista de "containers" (uno por tab) que se usan con `with`.
# El usuario navega haciendo clic; Streamlit re-ejecuta el script y renderiza la tab activa.
tab1, tab2, tab3 = st.tabs([
    "🔍 Comentario único",
    "📊 Análisis batch (CSV)",
    "💡 Insights accionables",
])

# Cada tab delega a su módulo correspondiente.
# Esto mantiene `app.py` mínimo y la lógica de cada vista contenida en su archivo.
# Pasamos el `pipeline` ya cargado para que los módulos no tengan que recargarlo.

with tab1:
    single_view.render(pipeline)

with tab2:
    batch_view.render(pipeline)

with tab3:
    # insights_view no usa el pipeline directamente (lee de session_state),
    # pero le pasamos el arg por consistencia.
    insights_view.render(pipeline)
