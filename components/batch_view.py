"""
================================================================================
 components/batch_view.py — TAB 2: Análisis batch de CSV
================================================================================
 Es el corazón de la app para uso en producción. Permite:

   1. Subir un CSV con reseñas (mínimo: columna 'texto'; opcional: 'sucursal', 'fecha')
   2. Validar el archivo (schema, encoding, filas mínimas)
   3. Procesar TODO el archivo en lote con una barra de progreso real
   4. Mostrar dashboard con KPIs, distribución, sucursales, evolución temporal
   5. Identificar reseñas para revisión humana (baja confianza)
   6. Descargar resultados enriquecidos (CSV) y reporte ejecutivo (PDF)

 Los resultados se persisten en st.session_state para que la Tab 3 (Insights)
 pueda usarlos sin volver a procesar.
================================================================================
"""

import io                                           # Para envolver bytes en un buffer leíble por pandas
import pandas as pd                                 # Manipulación de DataFrames
import streamlit as st                              # UI
from utils.viz import (                             # Gráficas reutilizables
    grafico_distribucion,
    grafico_sucursales,
    grafico_tiempo,
    grafico_confianza_distribucion,
    MAPA_COLORES,
)
from utils.report import generar_pdf                # Generador de PDF


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Columna obligatoria que el CSV DEBE tener para ser procesable.
# Si el usuario sube un archivo sin esta columna, mostramos error claro.
COLUMNA_REQUERIDA = 'texto'

# Columnas opcionales que, si existen, habilitan visualizaciones extra.
# - 'sucursal' habilita la gráfica de barras apiladas por local
# - 'fecha' habilita la serie temporal
COLUMNAS_OPCIONALES = ['sucursal', 'fecha']

# Umbral de confianza por debajo del cual marcamos para revisión humana.
# 0.75 viene del notebook original. En la UI lo dejamos editable con un slider.
UMBRAL_CONFIANZA_DEFAULT = 0.75

# Tamaño de batch para inferencia. Default razonable para CPU.
# El usuario podrá ajustarlo si tiene GPU (ver sidebar).
BATCH_SIZE_DEFAULT = 32


# ============================================================================
# CACHE: procesamiento del CSV
# ============================================================================
# @st.cache_data memoiza la función. Si el usuario sube el MISMO archivo dos veces
# (mismo hash de contenido), Streamlit retorna el resultado cacheado sin reprocesar.
# Esto es CLAVE en producción: con 1000 reseñas el procesamiento puede tardar 30s,
# y queremos que cambiar de tab no cause re-procesamiento.
#
# `show_spinner=False` porque queremos un spinner CUSTOM con barra de progreso
# (más informativo para el usuario que el spinner default de Streamlit).
@st.cache_data(show_spinner=False)
def _procesar_csv(contenido_bytes: bytes, _pipeline, batch_size: int) -> pd.DataFrame:
    """
    Procesa el CSV completo y retorna un DataFrame con las predicciones.

    Args:
        contenido_bytes: contenido del CSV como bytes (lo que retorna st.file_uploader.read()).
                         Lo recibimos como bytes (no DataFrame) porque bytes son hasheables
                         y el cache funciona — si pasáramos un DataFrame, fallaría el hash.
        _pipeline: instancia de PipelineSentimientos. Prefijo `_` le dice a Streamlit que
                   NO intente hashear este arg (los modelos PyTorch no son hasheables).
        batch_size: tamaño del batch de inferencia.

    Returns:
        DataFrame original + columnas: sentimiento, confianza, prob_*, idioma_detectado, traducido.
    """
    # Leer el CSV desde los bytes. io.BytesIO simula un archivo en memoria.
    # `encoding='utf-8'` cubre la mayoría de casos. Si falla, el caller debe re-encodear.
    # df = pd.read_csv(io.BytesIO(contenido_bytes))  ##CAMBIAR ENCODING

     # Leer el CSV con detección automática de encoding.
    # Problema: CSVs generados en Excel/Windows vienen en 'latin-1' o 'cp1252'
    # (no UTF-8), lo que rompe caracteres españoles (á, é, ñ, ü, etc.).
    # Solución: intentamos encodings en orden de probabilidad. Si el primero falla
    # (UnicodeDecodeError), probamos el siguiente.
    #
    # Orden de prueba:
    #   1. utf-8-sig: UTF-8 con BOM (Byte Order Mark). Excel guarda así cuando
    #      eliges "CSV UTF-8" — el BOM es un marcador invisible al inicio del archivo.
    #   2. utf-8: UTF-8 estándar (la mayoría de editores modernos).
    #   3. latin-1: ISO-8859-1, encoding histórico de Europa Occidental.
    #      Cubre á, é, í, ó, ú, ñ, ü perfectamente. Excel en español lo usa por default.
    #   4. cp1252: Windows-1252, superset de latin-1. Muy común en Windows en español.
    #
    # latin-1 es el fallback final porque acepta CUALQUIER byte 0-255 sin lanzar
    # UnicodeDecodeError — es imposible que falle.
    df = None
    for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(io.BytesIO(contenido_bytes), encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None:
        raise ValueError(
            "No se pudo leer el CSV. Intenta guardarlo como UTF-8 desde Excel: "
            "Archivo → Guardar como → CSV UTF-8 (delimitado por comas)."
        )

    # Extraer la columna de textos como lista. .fillna('') previene errores con celdas vacías.
    # .astype(str) garantiza que todo sea string (a veces pandas infiere int o float si las
    # primeras filas son numéricas, lo cual rompe el tokenizer).
    textos = df[COLUMNA_REQUERIDA].fillna('').astype(str).tolist()

    # Crear barra de progreso. st.progress(0) inicializa en 0%; luego la actualizamos.
    barra_progreso = st.progress(0, text="Iniciando análisis...")

    # Callback que se llama después de cada batch del pipeline.
    # Calcula el % completado y actualiza la barra.
    def actualizar_progreso(procesados: int, total: int):
        # division por cero defensiva; si total=0, %=0
        porcentaje = procesados / total if total > 0 else 0
        # st.progress acepta float [0,1]. El parámetro `text` muestra texto descriptivo.
        barra_progreso.progress(
            porcentaje,
            text=f"Analizando reseñas... {procesados:,} de {total:,} ({porcentaje*100:.0f}%)"
        )

    # Llamada al método batch del pipeline. Le pasamos el callback para que vaya
    # actualizando la barra mientras procesa.
    resultados = _pipeline.predecir_batch(
        textos,
        batch_size=batch_size,
        callback=actualizar_progreso,
    )

    # Cerrar la barra cuando termine (mostrar 100% por consistencia visual).
    barra_progreso.empty()

    # Convertimos la lista de dicts a DataFrame y la concatenamos con el original.
    # axis=1 = concatenar HORIZONTALMENTE (lado a lado). Así preservamos las columnas
    # del CSV original (id, fecha, sucursal, ...) y agregamos las del modelo.
    df_resultados = pd.DataFrame(resultados)
    df_combinado = pd.concat([df.reset_index(drop=True), df_resultados], axis=1)

    # Si hay columnas duplicadas tras el concat (ej: ambos tenían 'texto_original'),
    # nos quedamos con la última ocurrencia (la del modelo, que es la procesada).
    # `loc[:, ~duplicated]` filtra columnas duplicadas conservando la primera; usamos
    # take_last=True implícito para preferir las del modelo. Aquí simplificamos:
    df_combinado = df_combinado.loc[:, ~df_combinado.columns.duplicated(keep='last')]

    return df_combinado


# ============================================================================
# FUNCIÓN PRINCIPAL DE LA TAB
# ============================================================================

def render(pipeline):
    """
    Dibuja toda la UI de la Tab 2.

    Args:
        pipeline: instancia de PipelineSentimientos.
    """
    st.markdown("### Análisis batch de CSV")
    st.caption(
        "Sube un CSV con reseñas y obtén un dashboard completo. "
        f"El archivo debe tener una columna llamada `{COLUMNA_REQUERIDA}`. "
        f"Columnas opcionales para visualizaciones extra: {', '.join(COLUMNAS_OPCIONALES)}."
    )

    # ------------------------------------------------------------------
    # PASO 1: SUBIDA DEL ARCHIVO
    # ------------------------------------------------------------------
    archivo = st.file_uploader(
        "Selecciona tu archivo CSV",
        type=['csv'],                                   # solo permite CSVs (filtro del browser)
        help="UTF-8 recomendado. Tamaño máximo: 200MB (límite default de Streamlit).",
    )

    # Si no hay archivo, mostramos un placeholder con CSV de ejemplo descargable.
    # Esto ayuda a usuarios nuevos que no saben qué formato usar.
    if archivo is None:
        st.info("👆 Sube un archivo CSV para comenzar el análisis.")

        # Generamos un CSV de ejemplo en memoria para descarga.
        # Esto usa el mismo formato que la app espera, así el usuario puede usarlo como template.
        ejemplo_csv = _generar_csv_ejemplo()
        st.download_button(
            label="📥 Descargar CSV de ejemplo",
            data=ejemplo_csv,
            file_name="resenas_ejemplo.csv",
            mime="text/csv",
            help="Descarga este archivo, ábrelo en Excel/Sheets, llena con tus reseñas y vuelve a subirlo aquí.",
        )
        return  # Salimos: sin archivo no hay nada más que hacer.

    # ------------------------------------------------------------------
    # PASO 2: VALIDACIÓN DEL ARCHIVO
    # ------------------------------------------------------------------
    # Leemos el contenido como bytes. Lo guardamos para pasarlo al cache después.
    # archivo es un objeto UploadedFile de Streamlit, similar a un file handle.
    contenido_bytes = archivo.read()

    # Intentamos parsear el CSV para validarlo ANTES de procesar.
    # Si falla, mostramos error específico y abortamos.  ##CAMBIAR ENCODING
    # try:
    #     df_preview = pd.read_csv(io.BytesIO(contenido_bytes))
    # except Exception as e:
    #     st.error(f"❌ Error al leer el CSV: {e}")
    #     st.info("Verifica que el archivo sea un CSV válido en encoding UTF-8.")
    #     return
    
    df_preview = None
    for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            df_preview = pd.read_csv(io.BytesIO(contenido_bytes), encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df_preview is None:
        st.error("❌ No se pudo leer el CSV con ningún encoding conocido.")
        st.info(
            "Intenta guardarlo como UTF-8 desde Excel: "
            "Archivo → Guardar como → CSV UTF-8 (delimitado por comas)."
        )
        return

    # Intentamos parsear el CSV para validarlo ANTES de procesar.
    # Usamos la misma lógica de detección de encoding que _procesar_csv:
    # probamos utf-8-sig → utf-8 → latin-1 → cp1252 en orden.
    # Esto garantiza que el preview muestre correctamente tildes y ñ
    # independientemente de cómo el usuario generó el archivo.


    # Validar que exista la columna requerida.
    if COLUMNA_REQUERIDA not in df_preview.columns:
        st.error(
            f"❌ El CSV debe tener una columna llamada `{COLUMNA_REQUERIDA}`. "
            f"Columnas encontradas: {list(df_preview.columns)}"
        )
        return

    # Validar que tenga al menos algunas filas no vacías.
    filas_validas = df_preview[COLUMNA_REQUERIDA].dropna().astype(str).str.strip().str.len() > 0
    if filas_validas.sum() == 0:
        st.error("❌ La columna `texto` está vacía. No hay nada que analizar.")
        return

    # ------------------------------------------------------------------
    # PASO 3: PREVIEW + CONFIGURACIÓN
    # ------------------------------------------------------------------
    # Mostramos las primeras 5 filas como preview para que el usuario verifique
    # que el CSV se leyó correctamente antes de procesar.
    st.success(f"✅ Archivo cargado: **{archivo.name}** ({len(df_preview):,} filas)")

    with st.expander("👀 Ver vista previa de los datos", expanded=False):
        # st.dataframe es interactivo (sortable, scrollable). Mejor que st.table para datos.
        st.dataframe(df_preview.head(5), use_container_width=True)

        # Listamos qué columnas opcionales se detectaron, así el usuario sabe qué gráficas vendrán.
        cols_opcionales_detectadas = [c for c in COLUMNAS_OPCIONALES if c in df_preview.columns]
        if cols_opcionales_detectadas:
            st.info(
                f"📊 Columnas opcionales detectadas: {', '.join(cols_opcionales_detectadas)}. "
                f"Se generarán visualizaciones extra para estas dimensiones."
            )

    # Slider para el batch size (avanzado). Lo escondemos en un expander.
    with st.expander("⚙️ Configuración avanzada"):
        batch_size = st.slider(
            "Tamaño del batch (afecta velocidad)",
            min_value=8, max_value=128, value=BATCH_SIZE_DEFAULT, step=8,
            help="Más alto = más rápido pero usa más memoria. Si tienes GPU, sube a 64-128. "
                 "Si crashea, baja a 8-16.",
        )

    # ------------------------------------------------------------------
    # PASO 4: PROCESAR
    # ------------------------------------------------------------------
    # Botón principal. Solo aparece después de validar el archivo.
    if st.button("🚀 Procesar archivo", type="primary"):
        # Llamamos al wrapper cacheado. Si el archivo ya se procesó antes (mismo bytes),
        # esto retorna instantáneamente.
        df_resultados = _procesar_csv(contenido_bytes, pipeline, batch_size)

        # Guardamos en session_state para que la Tab 3 (Insights) pueda usarlo
        # sin reprocesar. session_state PERSISTE entre interacciones de la misma sesión.
        st.session_state.df_resultados = df_resultados
        st.session_state.archivo_nombre = archivo.name

        st.success(f"✅ ¡Análisis completado! {len(df_resultados):,} reseñas procesadas.")

    # ------------------------------------------------------------------
    # PASO 5: DASHBOARD (solo si ya hay resultados)
    # ------------------------------------------------------------------
    # Si ya procesamos (incluso en una interacción anterior), mostramos el dashboard.
    # Esto permite al usuario hacer scroll y volver al dashboard sin perderlo.
    if 'df_resultados' in st.session_state:
        _renderizar_dashboard(st.session_state.df_resultados, st.session_state.archivo_nombre)


# ============================================================================
# DASHBOARD COMPLETO (usado tras el procesamiento)
# ============================================================================

def _renderizar_dashboard(df: pd.DataFrame, nombre_archivo: str):
    """
    Renderiza el dashboard completo con KPIs, gráficas y tablas.

    Args:
        df: DataFrame con los resultados (incluye columnas del modelo).
        nombre_archivo: nombre del CSV original (para el header del PDF).
    """
    st.divider()
    st.markdown("## 📊 Resultados del análisis")

    # ---- BLOQUE 1: KPIs principales ----
    # 4 columnas con st.metric, igual al mockup que viste antes.
    # Filtramos errores para que las métricas reflejen solo predicciones válidas.
    df_validos = df[df['sentimiento'] != 'error']

    col1, col2, col3, col4 = st.columns(4)

    # KPI 1: Total
    col1.metric("Total reseñas", f"{len(df):,}")

    # KPIs 2-4: % por sentimiento
    # value_counts(normalize=True) da proporciones (suman 1.0). Multiplicamos por 100 → %.
    pcts = (df_validos['sentimiento'].value_counts(normalize=True) * 100).round(1)

    # .get() con default 0.0 maneja el caso donde una clase no aparece (ej: cero negativas).
    col2.metric("✅ Positivas", f"{pcts.get('positivo', 0.0):.1f}%")
    col3.metric("⚖️ Neutrales", f"{pcts.get('neutral', 0.0):.1f}%")
    col4.metric("❌ Negativas", f"{pcts.get('negativo', 0.0):.1f}%")

    # ---- BLOQUE 2: Distribución (donut) + Confianza (histograma) ----
    # Lado a lado para aprovechar el ancho de la pantalla.
    st.markdown("---")
    col_dist, col_conf = st.columns(2)

    with col_dist:
        st.markdown("##### Distribución de sentimientos")
        st.plotly_chart(grafico_distribucion(df_validos), use_container_width=True)

    with col_conf:
        st.markdown("##### Distribución de confianza")
        st.plotly_chart(grafico_confianza_distribucion(df_validos), use_container_width=True)

    # ---- BLOQUE 3: Por sucursal (si existe la columna) ----
    if 'sucursal' in df.columns:
        st.markdown("---")
        st.markdown("##### Sentimiento por sucursal")
        st.caption("Las sucursales con barras rojas largas requieren atención inmediata.")
        st.plotly_chart(grafico_sucursales(df_validos), use_container_width=True)

    # ---- BLOQUE 4: Evolución temporal (si existe fecha) ----
    if 'fecha' in df.columns:
        st.markdown("---")
        st.markdown("##### Evolución temporal")
        st.caption("Picos de negativas en fechas específicas suelen indicar problemas operativos puntuales.")
        st.plotly_chart(grafico_tiempo(df_validos), use_container_width=True)

    # ---- BLOQUE 5: Reseñas para revisión humana ----
    st.markdown("---")
    st.markdown("##### 🔍 Reseñas para revisión humana")

    # Slider para que el usuario ajuste el umbral interactivamente.
    umbral = st.slider(
        "Umbral de confianza (predicciones con confianza menor se marcan):",
        min_value=0.5, max_value=0.95,
        value=UMBRAL_CONFIANZA_DEFAULT,
        step=0.05,
        format="%.2f",  # mostrar 0.75 en lugar de 0.7500000000001
    )

    # Filtramos por confianza < umbral, ordenamos por confianza ASC (las más inciertas arriba).
    df_revision = df_validos[df_validos['confianza'] < umbral].sort_values('confianza')

    if len(df_revision) == 0:
        st.success(f"✨ ¡Excelente! Todas las predicciones tienen confianza ≥ {umbral:.0%}.")
    else:
        st.info(f"📋 {len(df_revision)} de {len(df_validos)} reseñas requieren revisión humana.")

        # Mostramos las 20 más inciertas en una tabla. .head(20) para no saturar la UI.
        # Seleccionamos solo las columnas relevantes (ocultando ruido como prob_*).
        cols_mostrar = ['texto_original', 'sentimiento', 'confianza']
        if 'sucursal' in df.columns:
            cols_mostrar.insert(1, 'sucursal')

        # st.dataframe con column_config permite formatear columnas (ej: % en lugar de decimal).
        st.dataframe(
            df_revision[cols_mostrar].head(20),
            use_container_width=True,
            hide_index=True,                                # oculta el índice numérico (más limpio)
            column_config={
                'texto_original': st.column_config.TextColumn('Reseña', width='large'),
                'sentimiento': st.column_config.TextColumn('Predicción', width='small'),
                'confianza': st.column_config.ProgressColumn(
                    'Confianza',
                    format="%.0f%%",
                    min_value=0.0, max_value=1.0,           # rango para la barra de progreso visual
                ),
            },
        )

    # ---- BLOQUE 6: Descargas ----
    st.markdown("---")
    st.markdown("##### 💾 Descargar resultados")

    col_csv, col_pdf = st.columns(2)

    # CSV: convertimos el DataFrame a bytes con .to_csv().encode('utf-8').
    # to_csv(index=False) omite el índice numérico (no aporta valor en archivos finales).
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    col_csv.download_button(
        label="📄 Descargar CSV completo",
        data=csv_bytes,
        file_name=f"resultados_{nombre_archivo}",
        mime="text/csv",
        use_container_width=True,
        help="CSV original + columnas del modelo (sentimiento, confianza, idioma, etc.)",
    )

    # PDF: generamos el reporte con la función centralizada y lo entregamos como bytes.
    # NOTA: la generación del PDF puede tardar 1-2 segundos. Por eso usamos un spinner.
    if col_pdf.button("📋 Generar reporte PDF", use_container_width=True):
        with st.spinner("Generando reporte..."):
            pdf_bytes = generar_pdf(df_validos, nombre_archivo=nombre_archivo)
        # Después de generar, mostramos el botón de descarga real.
        st.download_button(
            label="⬇️ Descargar reporte PDF",
            data=pdf_bytes,
            file_name=f"reporte_{nombre_archivo.replace('.csv', '')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ============================================================================
# CSV DE EJEMPLO (para usuarios que no tengan datos propios)
# ============================================================================

def _generar_csv_ejemplo() -> bytes:
    """
    Genera un CSV de ejemplo con 12 reseñas variadas en EN/ES.

    Las reseñas cubren positivas, negativas, neutrales, y ambiguas para que el usuario
    vea el rango completo de predicciones del modelo.

    Returns:
        bytes del CSV listos para st.download_button.
    """
    df_ejemplo = pd.DataFrame({
        'id': range(1, 13),
        'fecha': [
            '2024-11-01', '2024-11-02', '2024-11-03', '2024-11-04',
            '2024-11-05', '2024-11-06', '2024-11-07', '2024-11-08',
            '2024-11-09', '2024-11-10', '2024-11-11', '2024-11-12',
        ],
        'sucursal': [
            'Centro', 'Norte', 'Sur', 'Centro',
            'Norte', 'Sur', 'Centro', 'Norte',
            'Sur', 'Centro', 'Norte', 'Sur',
        ],
        'texto': [
            "La hamburguesa estaba increíble, súper jugosa. ¡Volveré pronto!",
            "Best fries I've ever had! Crispy outside, fluffy inside.",
            "Servicio muy lento, esperé más de 30 minutos. Comida regular.",
            "Cold burger, wrong order, rude staff. Terrible experience.",
            "Comida normal, nada del otro mundo pero tampoco mala.",
            "¡Excelente atención! El personal fue muy amable y la comida llegó rápida.",
            "Decent place, average food, average service.",
            "El lugar estaba sucio y las mesas no se limpiaron. No recomiendo.",
            "Amazing chicken sandwich! The sauce was perfect.",
            "Pedí sin cebolla y me trajeron con cebolla. Tuve que esperar.",
            "Muy buen precio para la cantidad de comida. Sabor decente.",
            "Worst customer service ever. The manager was extremely rude.",
        ],
    })
    return df_ejemplo.to_csv(index=False).encode('utf-8')
