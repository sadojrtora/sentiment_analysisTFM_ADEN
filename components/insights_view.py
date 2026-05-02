"""
================================================================================
 components/insights_view.py — TAB 3: Insights accionables
================================================================================
 Toma los resultados procesados en la Tab 2 y los traduce en INSIGHTS DE NEGOCIO.

 Mientras la Tab 2 te dice "QUÉ pasó" (cuántas negativas, en qué sucursal),
 esta tab responde "POR QUÉ y QUÉ HACER":

   1. Top n-gramas en reseñas negativas → identifica QUEJAS RECURRENTES
      (ej: "frío", "lento", "cebolla" se repiten = problema operativo concreto).

   2. Comparativo entre sucursales con alertas automáticas → identifica LOCALES PROBLEMA.

   3. Recomendaciones generadas con reglas simples sobre los datos.

 Esto es lo que diferencia un dashboard "que muestra datos" de uno "que ayuda a decidir".
================================================================================
"""

import streamlit as st                              # UI
import pandas as pd                                 # DataFrames
import re                                           # Limpieza adicional para el análisis de n-gramas
from collections import Counter                     # Conteo eficiente de palabras (más rápido que dict manual)
from sklearn.feature_extraction.text import CountVectorizer  # Vectorizador con stopwords integradas
from utils.viz import grafico_top_palabras, COLOR_NEGATIVO   # Gráfica reutilizable


# ============================================================================
# STOPWORDS — palabras a ignorar en el análisis de n-gramas
# ============================================================================
# Stopwords son palabras muy frecuentes que no aportan significado ("el", "la", "de").
# Si no las filtramos, dominarían el ranking de palabras frecuentes y ocultarían
# las palabras realmente informativas ("frío", "lento", etc.).
#
# Como las reseñas pueden estar en ES o EN (y nosotros las analizamos POST-traducción
# en inglés en algunos casos pero PRE-traducción en otros — usamos texto_original),
# combinamos stopwords de ambos idiomas. La traducción a inglés ocurre solo para el
# modelo, pero para insights de negocio queremos las palabras en el idioma del usuario.

STOPWORDS_ES = {
    # Artículos y determinantes
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
    # Preposiciones
    'de', 'del', 'a', 'al', 'en', 'con', 'por', 'para', 'sin', 'sobre',
    'entre', 'hasta', 'desde', 'hacia',
    # Conjunciones
    'y', 'o', 'pero', 'aunque', 'porque', 'como', 'que', 'si', 'ni',
    # Pronombres
    'yo', 'tu', 'el', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas',
    'me', 'te', 'se', 'le', 'lo', 'les', 'mi', 'tu', 'su',
    # Verbos auxiliares y ser/estar muy comunes
    'es', 'era', 'fue', 'son', 'sea', 'están', 'estaba', 'esta', 'estaban',
    'ha', 'he', 'has', 'han', 'habia', 'haber', 'hay',
    # Adverbios muy frecuentes
    'no', 'si', 'muy', 'mas', 'más', 'menos', 'tan', 'tanto', 'tambien', 'también',
    # Otros muy comunes en reseñas pero poco informativos
    'lugar', 'comida', 'restaurante', 'aqui', 'aquí', 'todo', 'todos', 'todas',
}

STOPWORDS_EN = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
    'my', 'your', 'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them',
    'it', 'its', 'as', 'so', 'if', 'than', 'then', 'just', 'too',
    # Comunes en reseñas pero poco informativos
    'food', 'place', 'restaurant', 'order', 'time', 'really',
}

# Unión de ambos sets para alimentar al CountVectorizer.
# CountVectorizer espera una lista (no un set), así que convertimos.
STOPWORDS_TODAS = list(STOPWORDS_ES | STOPWORDS_EN)


# ============================================================================
# UMBRALES DE NEGOCIO (para alertas y recomendaciones)
# ============================================================================
# Estos números son heurísticas razonables para el sector restaurantes.
# Un dueño puede ajustarlos según su contexto.

UMBRAL_NEG_ALERTA_GLOBAL = 25.0      # %neg > 25% → alerta global de calidad
UMBRAL_NEG_SUCURSAL_DELTA = 10.0     # sucursal con %neg > promedio + 10pp → alerta puntual
UMBRAL_CONFIANZA_BAJA = 0.70         # avg confianza < 70% → modelo dubitativo, considerar reentreno
TOP_N_PALABRAS = 12                  # cuántas palabras mostrar en el ranking


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def render(pipeline=None):
    """
    Dibuja la UI de la Tab 3.

    No usa el pipeline directamente (el procesamiento ya ocurrió en Tab 2);
    pero recibimos el arg por consistencia con las otras tabs.

    Args:
        pipeline: ignorado (acepta para mantener firma consistente con otras tabs).
    """
    st.markdown("### Insights accionables")
    st.caption(
        "Análisis profundo sobre los resultados procesados. "
        "Identifica patrones de quejas, sucursales problemáticas y prioridades de acción."
    )

    # Validación: necesitamos resultados de la Tab 2.
    # Si el usuario no procesó nada aún, mostramos un mensaje guía.
    if 'df_resultados' not in st.session_state:
        st.warning(
            "⚠️ Primero procesa un archivo en la pestaña **'Análisis batch (CSV)'**. "
            "Los insights se generan a partir de esos resultados."
        )
        return

    # Recuperamos el DataFrame procesado y filtramos errores.
    df = st.session_state.df_resultados
    df_validos = df[df['sentimiento'] != 'error'].copy()

    if len(df_validos) == 0:
        st.error("No hay reseñas válidas para analizar.")
        return

    # ------------------------------------------------------------------
    # SECCIÓN 1: Top palabras en reseñas NEGATIVAS
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🔍 ¿De qué se quejan los clientes?")
    st.caption(
        "Top palabras y bigramas (pares de palabras) más frecuentes en reseñas negativas. "
        "Las recurrencias suelen indicar problemas operativos concretos."
    )

    # Filtramos solo negativas. .copy() evita SettingWithCopyWarning de pandas.
    df_neg = df_validos[df_validos['sentimiento'] == 'negativo'].copy()

    if len(df_neg) < 5:
        # Con muy pocas reseñas, los n-gramas son ruido. Mejor avisar.
        st.info(
            f"Solo hay {len(df_neg)} reseña(s) negativa(s). "
            "Se necesitan al menos 5 para identificar patrones confiables."
        )
    else:
        # El usuario puede elegir entre palabras sueltas (unigrams) o pares (bigrams).
        # Los bigrams suelen ser más informativos: "servicio lento" > "servicio" o "lento" sueltos.
        ngram_seleccion = st.radio(
            "Tipo de análisis:",
            options=["Palabras individuales", "Pares de palabras (bigrams)"],
            horizontal=True,
            help="Bigrams capturan frases como 'mala atención' o 'comida fría', más accionables.",
        )

        # Mapeamos la selección del usuario a la tupla (min, max) que CountVectorizer entiende.
        # ngram_range=(1,1) = solo unigrams; (2,2) = solo bigrams.
        ngram_range = (1, 1) if ngram_seleccion == "Palabras individuales" else (2, 2)

        # Computamos los top n-gramas con la función auxiliar.
        top_palabras = _calcular_top_ngrams(
            textos=df_neg['texto_original'].tolist(),
            ngram_range=ngram_range,
            top_n=TOP_N_PALABRAS,
        )

        if top_palabras:
            # Mostramos la gráfica usando la función reutilizable de viz.py.
            # Color rojo = consistente con que son quejas.
            st.plotly_chart(
                grafico_top_palabras(top_palabras, color=COLOR_NEGATIVO),
                use_container_width=True,
            )

            # Insight automático basado en la palabra más frecuente.
            palabra_top, freq_top = top_palabras[0]
            pct_resenas_con_palabra = freq_top / len(df_neg) * 100
            st.info(
                f"💡 La palabra/frase más frecuente en quejas es **'{palabra_top}'**, "
                f"aparece en aproximadamente {pct_resenas_con_palabra:.0f}% de las reseñas negativas. "
                f"Considera investigar este patrón."
            )
        else:
            st.warning("No se pudieron extraer n-gramas con suficiente frecuencia.")

    # ------------------------------------------------------------------
    # SECCIÓN 2: Alertas por sucursal (si hay columna)
    # ------------------------------------------------------------------
    if 'sucursal' in df_validos.columns:
        st.markdown("---")
        st.markdown("#### 🚨 Alertas por sucursal")

        # Calculamos el % de negativas por sucursal.
        # Primero: total de reseñas por sucursal.
        total_por_sucursal = df_validos.groupby('sucursal').size()
        # Segundo: negativas por sucursal.
        neg_por_sucursal = df_validos[df_validos['sentimiento'] == 'negativo'].groupby('sucursal').size()
        # Tercero: alineamos índices y calculamos %. fillna(0) cubre sucursales sin negativas.
        pct_neg_sucursal = (neg_por_sucursal / total_por_sucursal * 100).fillna(0).sort_values(ascending=False)

        # Promedio global para usar como baseline.
        pct_neg_global = (df_validos['sentimiento'] == 'negativo').mean() * 100

        # Identificamos sucursales que superan el umbral (promedio + 10 puntos porcentuales).
        umbral_alerta = pct_neg_global + UMBRAL_NEG_SUCURSAL_DELTA
        sucursales_alerta = pct_neg_sucursal[pct_neg_sucursal > umbral_alerta]

        if len(sucursales_alerta) == 0:
            st.success(
                f"✅ Todas las sucursales están dentro del rango esperado "
                f"(promedio global: {pct_neg_global:.1f}% negativas)."
            )
        else:
            st.warning(
                f"⚠️ {len(sucursales_alerta)} sucursal(es) con desempeño por debajo del promedio. "
                f"Promedio global de negativas: {pct_neg_global:.1f}%. "
                f"Umbral de alerta: {umbral_alerta:.1f}%."
            )

            # Mostramos las sucursales en alerta como tarjetas/métricas.
            # Una columna por sucursal (max 4 visibles, las demás se truncan).
            cols = st.columns(min(len(sucursales_alerta), 4))
            for col, (sucursal, pct) in zip(cols, sucursales_alerta.head(4).items()):
                # delta muestra qué tanto está por encima del promedio. Negativo aquí
                # significa "peor que el promedio", por eso usamos delta_color="inverse".
                delta_pp = pct - pct_neg_global
                col.metric(
                    label=f"📍 {sucursal}",
                    value=f"{pct:.1f}%",
                    delta=f"+{delta_pp:.1f}pp vs promedio",
                    delta_color="inverse",  # rojo cuando aumenta (es malo aumentar negativas)
                )

        # Tabla resumen completa de TODAS las sucursales para contexto.
        with st.expander("📊 Ver tabla completa por sucursal"):
            tabla_resumen = pd.DataFrame({
                'Sucursal': pct_neg_sucursal.index,
                'Total reseñas': total_por_sucursal.reindex(pct_neg_sucursal.index).values,
                '% Negativas': pct_neg_sucursal.values.round(1),
            })
            st.dataframe(
                tabla_resumen,
                use_container_width=True,
                hide_index=True,
                column_config={
                    '% Negativas': st.column_config.ProgressColumn(
                        '% Negativas',
                        format="%.1f%%",
                        min_value=0.0, max_value=100.0,
                    ),
                },
            )

    # ------------------------------------------------------------------
    # SECCIÓN 3: Recomendaciones automáticas (resumen ejecutivo)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 💡 Recomendaciones")
    st.caption(
        "Sugerencias generadas automáticamente con reglas de negocio. "
        "Úsalas como punto de partida; siempre validar con tu equipo operativo."
    )

    recomendaciones = _generar_recomendaciones(df_validos)

    # Renderizamos cada recomendación como un st.info / st.warning / st.error
    # según severidad. Los íconos de color hacen el reporte mucho más escaneable.
    for severidad, texto in recomendaciones:
        if severidad == 'critica':
            st.error(f"🔴 {texto}")
        elif severidad == 'advertencia':
            st.warning(f"🟡 {texto}")
        elif severidad == 'info':
            st.info(f"🔵 {texto}")
        else:  # 'ok'
            st.success(f"🟢 {texto}")


# ============================================================================
# HELPERS
# ============================================================================

def _calcular_top_ngrams(textos: list, ngram_range: tuple, top_n: int) -> list:
    """
    Calcula los n-gramas (palabras o pares) más frecuentes en una lista de textos.

    Usamos sklearn CountVectorizer porque:
      - Tokeniza eficientemente (regex optimizado en C)
      - Maneja stopwords sin código adicional
      - Soporta n-gramas de cualquier tamaño con ngram_range
      - Es ~50x más rápido que un loop manual con regex

    Args:
        textos: lista de strings a analizar.
        ngram_range: tupla (min, max). (1,1)=palabras, (2,2)=bigrams, (1,2)=ambos.
        top_n: cuántos términos retornar (los más frecuentes).

    Returns:
        Lista de tuplas (término, frecuencia) ordenada DESC por frecuencia.
    """
    # Validación: si no hay textos, retornamos lista vacía (la UI lo manejará).
    if not textos:
        return []

    # Pre-limpieza: bajar a minúsculas y eliminar caracteres especiales no-letras.
    # Esto ayuda a unificar "Servicio" y "servicio" como el mismo término.
    textos_limpios = [
        re.sub(r'[^a-záéíóúüñ\s]', ' ', t.lower()) for t in textos
    ]

    try:
        # CountVectorizer:
        #   - ngram_range: qué tamaño de n-gramas extraer
        #   - stop_words: lista de palabras a ignorar
        #   - min_df=2: el término debe aparecer en al menos 2 documentos para ser considerado
        #               (filtra ruido único — palabras que solo aparecen una vez no son patrones)
        #   - max_features=200: cap superior para no consumir memoria con vocabularios enormes
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=STOPWORDS_TODAS,
            min_df=2,
            max_features=200,
        )

        # fit_transform aprende el vocabulario y devuelve la matriz documento-término.
        # Es una matriz sparse (mayoría de ceros): (n_docs, n_features).
        matriz = vectorizer.fit_transform(textos_limpios)

        # Sumamos por columnas → frecuencia total de cada n-grama en TODO el corpus.
        # .sum(axis=0) suma a lo largo de las filas (documentos). asarray + ravel los aplana a 1D.
        frecuencias = matriz.sum(axis=0).A1  # .A1 convierte matriz sparse → array 1D denso

        # Obtenemos los nombres de los features (n-gramas) en el mismo orden.
        # zip los empareja con sus frecuencias.
        terminos = vectorizer.get_feature_names_out()

        # Ordenamos por frecuencia DESC y tomamos los top_n.
        pares = list(zip(terminos, frecuencias))
        pares_ordenados = sorted(pares, key=lambda x: x[1], reverse=True)[:top_n]

        # Convertimos numpy ints a Python ints para evitar problemas de serialización.
        return [(termino, int(freq)) for termino, freq in pares_ordenados]

    except ValueError:
        # CountVectorizer lanza ValueError si todos los términos son stopwords (vocabulario vacío).
        # En ese caso retornamos lista vacía para que la UI muestre un mensaje apropiado.
        return []


def _generar_recomendaciones(df: pd.DataFrame) -> list:
    """
    Genera recomendaciones basadas en reglas de negocio sobre los datos.

    Cada recomendación es una tupla (severidad, texto). Severidades:
      - 'critica':    🔴 — requiere acción inmediata
      - 'advertencia': 🟡 — requiere atención
      - 'info':       🔵 — observación útil
      - 'ok':         🟢 — todo bien

    Args:
        df: DataFrame de resultados válidos.

    Returns:
        Lista de tuplas (severidad, texto), ordenadas por severidad descendente.
    """
    recs = []

    # --- Métricas base ---
    pct_neg = (df['sentimiento'] == 'negativo').mean() * 100
    pct_pos = (df['sentimiento'] == 'positivo').mean() * 100
    confianza_prom = df['confianza'].mean()

    # --- Regla 1: % de negativas global ---
    if pct_neg > UMBRAL_NEG_ALERTA_GLOBAL:
        recs.append((
            'critica',
            f"**Tasa de negativas elevada ({pct_neg:.1f}%)**. Esto está por encima de "
            f"{UMBRAL_NEG_ALERTA_GLOBAL:.0f}%, considerado el umbral de alerta para el sector. "
            f"Recomendación: auditoría operativa urgente (calidad de comida, tiempos de servicio, "
            f"limpieza, capacitación de personal)."
        ))
    elif pct_neg > 15:
        recs.append((
            'advertencia',
            f"**{pct_neg:.1f}% de negativas requiere monitoreo.** Identifica los patrones "
            f"recurrentes en la sección de palabras frecuentes y prioriza los más "
            f"mencionados."
        ))

    # --- Regla 2: dominancia de positivas (señal de fortaleza) ---
    if pct_pos > 70:
        recs.append((
            'ok',
            f"**Excelente desempeño global ({pct_pos:.1f}% positivas).** Identifica qué "
            f"prácticas están funcionando y replícalas en sucursales con menor desempeño."
        ))

    # --- Regla 3: confianza del modelo ---
    if confianza_prom < UMBRAL_CONFIANZA_BAJA:
        recs.append((
            'info',
            f"**Confianza promedio del modelo baja ({confianza_prom*100:.1f}%).** Muchas "
            f"reseñas son ambiguas. Considera reentrenar el modelo con datos del dominio "
            f"específico, o ajustar el umbral de revisión humana."
        ))

    # --- Regla 4: alertas por sucursal (si la columna existe) ---
    if 'sucursal' in df.columns:
        total_sucursal = df.groupby('sucursal').size()
        neg_sucursal = df[df['sentimiento'] == 'negativo'].groupby('sucursal').size()
        pct_neg_sucursal = (neg_sucursal / total_sucursal * 100).fillna(0)

        umbral = pct_neg + UMBRAL_NEG_SUCURSAL_DELTA
        peores = pct_neg_sucursal[pct_neg_sucursal > umbral].sort_values(ascending=False)

        if len(peores) > 0:
            sucursales_lista = ', '.join([f"**{s}** ({v:.0f}%)" for s, v in peores.items()])
            recs.append((
                'advertencia',
                f"**Sucursales con desempeño debajo del promedio:** {sucursales_lista}. "
                f"El promedio global de negativas es {pct_neg:.1f}%. Evaluar capacitación "
                f"o intervención operativa en estos locales específicamente."
            ))

    # --- Regla 5: volumen de datos suficiente ---
    if len(df) < 50:
        recs.append((
            'info',
            f"**Volumen de datos limitado ({len(df)} reseñas).** Las conclusiones son "
            f"orientativas. Acumula más reseñas (idealmente >100) para análisis más confiables."
        ))

    # --- Regla 6: si no hay nada que reportar, mensaje positivo ---
    if not recs:
        recs.append((
            'ok',
            "**Todos los indicadores están en rangos saludables.** Mantén el monitoreo "
            "regular ejecutando este análisis mensualmente."
        ))

    return recs
