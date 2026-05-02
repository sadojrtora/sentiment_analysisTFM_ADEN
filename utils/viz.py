"""
================================================================================
 utils/viz.py — Funciones de visualización reutilizables (Plotly)
================================================================================
 Centralizamos aquí TODAS las funciones de generación de gráficas para:
   1. Mantener la paleta de colores consistente entre tabs
   2. Evitar duplicar código (DRY: Don't Repeat Yourself)
   3. Poder cambiar el estilo en un solo lugar y que se actualice toda la app

 Usamos Plotly (en vez de matplotlib) porque:
   - Las gráficas son INTERACTIVAS por default (hover, zoom, pan, descarga)
   - Streamlit las renderiza nativamente con st.plotly_chart()
   - Mejor calidad visual sin tener que tunear nada
================================================================================
"""

# Plotly Express: API de alto nivel, sintaxis tipo seaborn (una línea por gráfica).
# Útil para gráficas estándar (pie, bar, line, etc.) sin configuración extensa.
import plotly.express as px

# Plotly Graph Objects: API de bajo nivel, control total. La usamos cuando express no alcanza
# (ej: barras horizontales con colores específicos por categoría).
import plotly.graph_objects as go

import pandas as pd  # DataFrames para manipular datos antes de graficar


# ============================================================================
# PALETA DE COLORES — usada en TODA la app
# ============================================================================
# Estos colores vienen del notebook original y se eligieron por semántica:
# verde = positivo, naranja = neutral, rojo = negativo (intuición universal).
# Mantenerlos aquí permite cambiarlos en un solo lugar.

COLOR_POSITIVO = '#1D9E75'   # Teal/verde — emociones positivas, OK
COLOR_NEUTRAL = '#BA7517'    # Ámbar/naranja — atención requerida pero no crítico
COLOR_NEGATIVO = '#E24B4A'   # Rojo — alerta, acción requerida

# Diccionario que mapea sentimiento → color. Plotly lo usa con `color_discrete_map`
# para asignar colores consistentes sin que importe el orden de las categorías.
MAPA_COLORES = {
    'positivo': COLOR_POSITIVO,
    'neutral': COLOR_NEUTRAL,
    'negativo': COLOR_NEGATIVO,
    'error': '#999999',  # Gris neutro para textos que no pudieron procesarse
}

# Orden lógico para que las leyendas y barras siempre aparezcan así:
# negativo → neutral → positivo (tipo escala Likert, de peor a mejor).
ORDEN_SENTIMIENTOS = ['negativo', 'neutral', 'positivo']


# ============================================================================
# GRÁFICAS — predicción individual (Tab 1)
# ============================================================================

def grafico_probabilidades_single(probabilidades: dict) -> go.Figure:
    """
    Barras horizontales con la probabilidad de cada clase para una sola reseña.

    Útil en Tab 1 para mostrar al usuario CUÁN seguro está el modelo de cada opción,
    no solo la clase ganadora. Por ejemplo: si las probs son 0.45/0.30/0.25, el modelo
    eligió "positivo" pero con baja confianza — la UI debe destacarlo.

    Args:
        probabilidades: dict {'positivo': 0.85, 'neutral': 0.10, 'negativo': 0.05}

    Returns:
        Figura de Plotly lista para st.plotly_chart().
    """
    # Convertimos el dict a listas en el orden lógico que definimos arriba.
    # Esto garantiza que la barra "negativo" siempre quede arriba, "positivo" abajo.
    sentimientos = ORDEN_SENTIMIENTOS
    valores = [probabilidades[s] for s in sentimientos]
    colores = [MAPA_COLORES[s] for s in sentimientos]

    # go.Figure con un solo trace de barras horizontales (orientation='h').
    figura = go.Figure(
        data=[go.Bar(
            x=valores,                                          # eje X = probabilidades (0 a 1)
            y=sentimientos,                                     # eje Y = nombres de las clases
            orientation='h',                                    # barras horizontales (más legible)
            marker=dict(color=colores),                         # color por barra (ver MAPA_COLORES)
            text=[f"{v*100:.1f}%" for v in valores],            # etiquetas: "85.3%" en cada barra
            textposition='outside',                             # texto fuera de la barra (más limpio)
            textfont=dict(size=14),
            hovertemplate='%{y}: %{x:.1%}<extra></extra>',      # tooltip al hacer hover
        )]
    )

    # Layout: márgenes mínimos, fondo transparente para integrarse con Streamlit.
    figura.update_layout(
        xaxis=dict(
            range=[0, 1.15],                                    # un poco de margen para que quepan los labels
            tickformat='.0%',                                   # eje en porcentaje (0%, 25%, 50%...)
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
        ),
        yaxis=dict(showgrid=False),
        margin=dict(l=80, r=20, t=20, b=40),                    # margen izquierdo más ancho para los labels
        height=200,                                              # compacto, no domina la página
        showlegend=False,                                        # no hace falta, los colores se entienden solos
        plot_bgcolor='rgba(0,0,0,0)',                            # fondo transparente
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return figura


# ============================================================================
# GRÁFICAS — análisis batch (Tab 2)
# ============================================================================

def grafico_distribucion(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart con la distribución de sentimientos en el dataset.

    Por qué donut (no pie completo): el agujero central deja espacio para mostrar
    el TOTAL en grande (anotación), que es la métrica más importante de un vistazo.

    Args:
        df: DataFrame con columna 'sentimiento'.

    Returns:
        Figura Plotly tipo donut.
    """
    # value_counts() cuenta ocurrencias de cada categoría.
    # .reindex() asegura que estén en el orden lógico y que las clases ausentes aparezcan en 0.
    conteo = df['sentimiento'].value_counts().reindex(ORDEN_SENTIMIENTOS, fill_value=0)

    figura = go.Figure(
        data=[go.Pie(
            labels=conteo.index.tolist(),                       # nombres de las clases
            values=conteo.values.tolist(),                      # cantidad por clase
            hole=0.55,                                          # 0.55 = donut grueso pero con buen agujero central
            marker=dict(colors=[MAPA_COLORES[s] for s in conteo.index]),
            textinfo='percent',                                 # mostrar % dentro de cada slice
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>%{value} reseñas (%{percent})<extra></extra>',
        )]
    )

    # Anotación central: el total grande en el medio del donut.
    # Esto es lo que diferencia un donut profesional de uno default feo.
    figura.add_annotation(
        text=f"<b>{len(df):,}</b><br><span style='font-size:12px'>reseñas</span>",
        x=0.5, y=0.5,                                           # centro del donut
        font=dict(size=24),
        showarrow=False,
    )

    figura.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        showlegend=True,
        legend=dict(orientation='h', y=-0.1),                   # leyenda horizontal abajo (ahorra espacio)
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return figura


def grafico_sucursales(df: pd.DataFrame, columna_sucursal: str = 'sucursal') -> go.Figure:
    """
    Barras apiladas mostrando % de cada sentimiento POR sucursal.

    Es la gráfica más accionable: identifica RÁPIDO qué local tiene más quejas.
    Usamos % en lugar de conteos absolutos porque las sucursales pueden tener
    volúmenes muy distintos (Centro=200 reseñas, Sur=50) y queremos comparar PROPORCIÓN.

    Args:
        df: DataFrame con columnas 'sentimiento' y la de sucursal.
        columna_sucursal: nombre de la columna que identifica la sucursal.

    Returns:
        Figura Plotly de barras apiladas al 100%.
    """
    # Contingencia (tabla cruzada): cuenta combinaciones sucursal × sentimiento.
    # Resultado: filas=sucursales, columnas=sentimientos, valores=conteo.
    tabla = pd.crosstab(df[columna_sucursal], df['sentimiento'])

    # Aseguramos que existan las 3 columnas aunque alguna clase no aparezca en los datos.
    # reindex(columns=...) llena con 0 las que falten.
    tabla = tabla.reindex(columns=ORDEN_SENTIMIENTOS, fill_value=0)

    # Convertir a porcentajes por fila (cada sucursal suma 100%).
    # axis=0 normaliza por columna; axis=1 por fila. Queremos por fila (sucursal).
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

    # Construir un trace por sentimiento (cada uno es una "capa" de la barra apilada).
    figura = go.Figure()
    for sentimiento in ORDEN_SENTIMIENTOS:
        figura.add_trace(go.Bar(
            name=sentimiento,                                   # nombre en la leyenda
            y=tabla_pct.index.tolist(),                         # eje Y = sucursales (horizontal)
            x=tabla_pct[sentimiento].tolist(),                  # eje X = porcentaje
            orientation='h',
            marker=dict(color=MAPA_COLORES[sentimiento]),
            # Texto que va dentro de cada segmento de barra (solo si la barra es grande).
            text=[f"{v:.0f}%" if v >= 8 else "" for v in tabla_pct[sentimiento]],
            textposition='inside',
            textfont=dict(color='white', size=11),
            hovertemplate=(
                f"<b>%{{y}}</b><br>"
                f"{sentimiento}: %{{x:.1f}}%<br>"
                f"({{customdata}} reseñas)<extra></extra>"
            ).replace('{customdata}', '%{customdata}'),
            customdata=tabla[sentimiento].tolist(),             # conteo absoluto en el tooltip
        ))

    figura.update_layout(
        barmode='stack',                                        # CLAVE: apilar las barras
        xaxis=dict(
            title='% de reseñas',
            range=[0, 100],
            ticksuffix='%',                                     # mostrar "80%" en lugar de "80"
        ),
        yaxis=dict(title='', autorange='reversed'),             # primer registro arriba (orden natural de lectura)
        margin=dict(l=20, r=20, t=20, b=40),
        height=max(150, len(tabla_pct) * 50),                   # altura adaptativa según # de sucursales
        legend=dict(orientation='h', y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return figura


def grafico_tiempo(df: pd.DataFrame, columna_fecha: str = 'fecha') -> go.Figure:
    """
    Serie temporal: # de reseñas por día agrupado por sentimiento.

    Útil para detectar TENDENCIAS y EVENTOS: un pico de negativas el 15 de noviembre
    podría indicar un problema operativo ese día (ingrediente vencido, falta de personal).

    Args:
        df: DataFrame con columna de fecha y 'sentimiento'.
        columna_fecha: nombre de la columna de fecha.

    Returns:
        Figura Plotly de líneas con un trace por sentimiento.
    """
    # Convertir a datetime de forma robusta. errors='coerce' convierte fechas mal
    # parseadas a NaT (Not a Time) en lugar de lanzar excepción.
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')

    # Agrupar por (fecha, sentimiento) y contar. unstack() convierte las clases en columnas.
    # fill_value=0 llena los días donde alguna clase no tuvo reseñas.
    serie = df.groupby([columna_fecha, 'sentimiento']).size().unstack(fill_value=0)
    serie = serie.reindex(columns=ORDEN_SENTIMIENTOS, fill_value=0)

    # Construir un trace por sentimiento. Líneas con relleno bajo la curva (tozeroy)
    # hace que la gráfica se sienta más "compacta" y menos vacía.
    figura = go.Figure()
    for sentimiento in ORDEN_SENTIMIENTOS:
        figura.add_trace(go.Scatter(
            x=serie.index,                                      # fechas
            y=serie[sentimiento],                               # conteo de reseñas
            mode='lines+markers',
            name=sentimiento,
            line=dict(color=MAPA_COLORES[sentimiento], width=2),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%d %b %Y}</b><br>' + sentimiento + ': %{y}<extra></extra>',
        ))

    figura.update_layout(
        xaxis=dict(title='Fecha', showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title='# reseñas', showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        margin=dict(l=40, r=20, t=20, b=40),
        height=320,
        hovermode='x unified',                                  # tooltip unificado: muestra los 3 valores en una fecha
        legend=dict(orientation='h', y=-0.15),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return figura


# ============================================================================
# GRÁFICAS — insights (Tab 3)
# ============================================================================

def grafico_top_palabras(palabras_freq: list, color: str = COLOR_NEGATIVO) -> go.Figure:
    """
    Barras horizontales con las palabras/n-gramas más frecuentes.

    Usado en Tab 3 para mostrar las palabras que más aparecen en reseñas negativas.
    Es la herramienta clave para detectar quejas recurrentes ("frío", "lento", "sucio").

    Args:
        palabras_freq: lista de tuplas (palabra, frecuencia) ordenadas DESC.
        color: color de las barras (default: rojo, indicativo de quejas).

    Returns:
        Figura Plotly de barras horizontales.
    """
    # Desempacamos las tuplas en dos listas paralelas.
    # Invertimos el orden ([::-1]) para que la palabra más frecuente quede ARRIBA en la gráfica
    # (Plotly dibuja el primer elemento del array abajo del eje Y por default).
    palabras = [p for p, _ in palabras_freq][::-1]
    frecuencias = [f for _, f in palabras_freq][::-1]

    figura = go.Figure(
        data=[go.Bar(
            x=frecuencias,
            y=palabras,
            orientation='h',
            marker=dict(color=color),
            text=frecuencias,                                   # mostrar el conteo al final de cada barra
            textposition='outside',
            hovertemplate='<b>%{y}</b>: %{x} apariciones<extra></extra>',
        )]
    )

    figura.update_layout(
        xaxis=dict(title='Frecuencia', showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title=''),
        margin=dict(l=120, r=40, t=20, b=40),                   # margen izq amplio para palabras largas
        # Altura proporcional al número de palabras (mínimo 200px para que no se vea apretado).
        height=max(200, len(palabras) * 28),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return figura


def grafico_confianza_distribucion(df: pd.DataFrame) -> go.Figure:
    """
    Histograma de la distribución de confianzas del modelo.

    Si la mayoría de predicciones tiene confianza >0.9 → modelo muy seguro, dataset claro.
    Si hay muchas en 0.4-0.7 → muchas reseñas ambiguas, considerar revisión humana.

    Args:
        df: DataFrame con columna 'confianza'.

    Returns:
        Figura Plotly tipo histograma.
    """
    # px.histogram: API rápida de Plotly Express, perfecta para distribuciones.
    figura = px.histogram(
        df,
        x='confianza',
        nbins=20,                                               # 20 bins = buena granularidad sin ruido
        color_discrete_sequence=['#3B6D11'],                    # verde oscuro neutro
    )

    # Línea vertical en 0.75 = umbral típico para "alta confianza".
    # add_vline es atajo de Plotly para anotaciones verticales.
    figura.add_vline(
        x=0.75,
        line_dash='dash',
        line_color=COLOR_NEGATIVO,
        annotation_text='Umbral revisión humana',
        annotation_position='top',
    )

    figura.update_layout(
        xaxis=dict(title='Confianza del modelo', tickformat='.0%', range=[0, 1]),
        yaxis=dict(title='# reseñas'),
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
        showlegend=False,
        bargap=0.05,                                            # pequeño espacio entre barras (más limpio)
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return figura
