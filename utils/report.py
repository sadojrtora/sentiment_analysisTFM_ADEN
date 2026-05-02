"""
================================================================================
 utils/report.py — Generador de reporte ejecutivo en PDF
================================================================================
 Toma el DataFrame de resultados y produce un PDF de 1-2 páginas con:
   - Header con fecha y nombre del archivo procesado
   - KPIs principales (total, % por sentimiento, confianza promedio)
   - Tabla resumen por sucursal (si existe)
   - Top 5 reseñas negativas (las más críticas)
   - Recomendaciones automáticas

 Usamos `reportlab` porque:
   - Es el estándar de facto para PDFs en Python
   - No requiere wkhtmltopdf ni navegador (todo en Python puro)
   - Permite control PIXEL-PERFECT del layout
================================================================================
"""

# --- ReportLab imports ---
# `letter` es el tamaño de página estándar US (8.5x11 in). Para EU/LatAm también funciona bien.
from reportlab.lib.pagesizes import letter
# Tabla y estilos para datos tabulares.
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# `getSampleStyleSheet` da un set de estilos predefinidos (Title, Heading1, BodyText, ...).
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# Constantes de unidades. inch = 72 pts. Útil para márgenes legibles.
from reportlab.lib.units import inch
# Colors: paleta predefinida (red, green, etc.) y método HexColor para colores custom.
from reportlab.lib import colors
# Alineaciones del texto en los Paragraphs.
from reportlab.lib.enums import TA_LEFT, TA_CENTER

import io                # Para generar el PDF en memoria (BytesIO) y enviarlo a Streamlit
from datetime import datetime  # Para timestamp en el header
import pandas as pd


# Colores corporativos coherentes con el resto de la app (matching utils/viz.py).
# reportlab usa su propio formato de color, por eso convertimos con HexColor().
COLOR_POSITIVO = colors.HexColor('#1D9E75')
COLOR_NEUTRAL = colors.HexColor('#BA7517')
COLOR_NEGATIVO = colors.HexColor('#E24B4A')
COLOR_GRIS_CLARO = colors.HexColor('#F5F5F5')
COLOR_GRIS_OSCURO = colors.HexColor('#333333')


def generar_pdf(df: pd.DataFrame, nombre_archivo: str = 'reseñas.csv') -> bytes:
    """
    Genera un PDF ejecutivo a partir del DataFrame de resultados de análisis.

    Args:
        df: DataFrame con columnas 'sentimiento', 'confianza', 'texto_original',
            y opcionalmente 'sucursal', 'fecha'.
        nombre_archivo: nombre del CSV original para mostrar en el header.

    Returns:
        bytes con el PDF binario, listo para st.download_button() o guardar a disco.
    """
    # BytesIO actúa como un "archivo en memoria". Más eficiente que escribir a disco
    # y luego leerlo de vuelta. Streamlit puede mandar estos bytes directo al navegador.
    buffer = io.BytesIO()

    # SimpleDocTemplate es el contenedor principal del documento.
    # Definimos márgenes generosos (0.75 inch ≈ 1.9 cm) para look profesional.
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    # `story` es la lista de elementos que se "fluirán" en las páginas.
    # ReportLab los acomoda automáticamente, saltando de página cuando hace falta.
    story = []

    # Cargamos los estilos predefinidos como base, luego customizamos algunos.
    estilos = getSampleStyleSheet()

    # Estilo custom para el título principal: grande, alineado a la izquierda, color oscuro.
    estilo_titulo = ParagraphStyle(
        'TituloCustom',
        parent=estilos['Title'],
        fontSize=22,
        alignment=TA_LEFT,
        textColor=COLOR_GRIS_OSCURO,
        spaceAfter=4,
    )

    # Estilo para el subtítulo (fecha + archivo).
    estilo_subtitulo = ParagraphStyle(
        'SubtituloCustom',
        parent=estilos['BodyText'],
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=20,
    )

    # Estilo para los headings de sección.
    estilo_seccion = ParagraphStyle(
        'SeccionCustom',
        parent=estilos['Heading2'],
        fontSize=14,
        textColor=COLOR_GRIS_OSCURO,
        spaceBefore=14,
        spaceAfter=8,
    )

    # ------------------------------------------------------------------
    # SECCIÓN 1: HEADER
    # ------------------------------------------------------------------
    story.append(Paragraph('Reporte de análisis de sentimientos', estilo_titulo))
    fecha_actual = datetime.now().strftime('%d de %B de %Y, %H:%M')
    story.append(Paragraph(
        f'Archivo: <b>{nombre_archivo}</b> · Generado: {fecha_actual}',
        estilo_subtitulo,
    ))

    # ------------------------------------------------------------------
    # SECCIÓN 2: KPIs PRINCIPALES (tabla 2x4 con métricas clave)
    # ------------------------------------------------------------------
    story.append(Paragraph('Métricas principales', estilo_seccion))

    total = len(df)
    # Calculamos % por sentimiento. Multiplicamos por 100 y redondeamos a entero.
    # Si una clase no aparece, .get() retorna 0.0 en lugar de KeyError.
    pct_por_sent = (df['sentimiento'].value_counts(normalize=True) * 100).round(1)
    pct_pos = pct_por_sent.get('positivo', 0.0)
    pct_neu = pct_por_sent.get('neutral', 0.0)
    pct_neg = pct_por_sent.get('negativo', 0.0)
    confianza_prom = df['confianza'].mean() * 100  # promedio en %

    # Construimos la tabla como lista de listas (cada sublista es una fila).
    # Fila 1: labels. Fila 2: valores. Esto se ve como una tabla de 4 KPIs lado a lado.
    datos_kpis = [
        ['Total reseñas', '% positivas', '% neutrales', '% negativas'],
        [f'{total:,}', f'{pct_pos:.1f}%', f'{pct_neu:.1f}%', f'{pct_neg:.1f}%'],
    ]

    # Creamos la tabla. colWidths reparte el ancho disponible entre las 4 columnas.
    tabla_kpis = Table(datos_kpis, colWidths=[1.5 * inch] * 4)

    # TableStyle: aplica formato a celdas específicas usando coordenadas (col, fila).
    # (-1, -1) significa "última columna, última fila". (0, 0) es la esquina sup-izq.
    tabla_kpis.setStyle(TableStyle([
        # Fila 0 (headers): fondo gris, texto blanco, centrado, padding generoso.
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_GRIS_OSCURO),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        # Fila 1 (valores): fondo gris claro, texto grande para destacar.
        ('BACKGROUND', (0, 1), (-1, 1), COLOR_GRIS_CLARO),
        ('FONTSIZE', (0, 1), (-1, 1), 16),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (-1, 1), COLOR_GRIS_OSCURO),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
        ('TOPPADDING', (0, 1), (-1, 1), 12),
        # Color específico por columna en la fila 1 (el % de cada sentimiento con su color).
        ('TEXTCOLOR', (1, 1), (1, 1), COLOR_POSITIVO),  # columna 1 fila 1 = % positivas
        ('TEXTCOLOR', (2, 1), (2, 1), COLOR_NEUTRAL),
        ('TEXTCOLOR', (3, 1), (3, 1), COLOR_NEGATIVO),
    ]))
    story.append(tabla_kpis)

    # KPI extra: confianza promedio (en su propio párrafo para no saturar la tabla).
    story.append(Spacer(1, 12))  # Spacer = espacio en blanco vertical (12 pts)
    story.append(Paragraph(
        f'<b>Confianza promedio del modelo:</b> {confianza_prom:.1f}%',
        estilos['BodyText'],
    ))

    # ------------------------------------------------------------------
    # SECCIÓN 3: BREAKDOWN POR SUCURSAL (si existe la columna)
    # ------------------------------------------------------------------
    if 'sucursal' in df.columns:
        story.append(Paragraph('Desglose por sucursal', estilo_seccion))

        # Generamos una tabla cruzada de % por sucursal y sentimiento.
        # margins=False (que es el default de pd.crosstab; no queremos totales aquí).
        tabla_pct = pd.crosstab(
            df['sucursal'],
            df['sentimiento'],
            normalize='index',   # normaliza por fila (cada sucursal suma 100%)
        ) * 100

        # Headers de la tabla. Empezamos con "Sucursal" y agregamos las columnas existentes.
        headers = ['Sucursal'] + list(tabla_pct.columns)
        # Filas de datos: nombre de sucursal + valores formateados como "85.0%".
        # iterrows() itera (índice, fila) — el índice es el nombre de la sucursal.
        filas_data = []
        for sucursal, fila in tabla_pct.iterrows():
            filas_data.append([sucursal] + [f'{v:.1f}%' for v in fila.values])

        datos_sucursal = [headers] + filas_data

        # Tabla con ancho proporcional al # de columnas.
        ancho_col = 5.5 * inch / len(headers)
        tabla_sucursal = Table(datos_sucursal, colWidths=[ancho_col] * len(headers))
        tabla_sucursal.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_GRIS_OSCURO),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # bordes de 0.5pt en todas las celdas
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(tabla_sucursal)

    # ------------------------------------------------------------------
    # SECCIÓN 4: TOP 5 RESEÑAS NEGATIVAS (las más urgentes)
    # ------------------------------------------------------------------
    story.append(Paragraph('Reseñas negativas más relevantes', estilo_seccion))

    # Filtramos negativas y ordenamos por confianza DESC (las que el modelo está más seguro
    # que son negativas → las más críticas y representativas).
    df_neg = df[df['sentimiento'] == 'negativo'].sort_values('confianza', ascending=False).head(5)

    if len(df_neg) == 0:
        story.append(Paragraph(
            '<i>No se detectaron reseñas negativas en este lote — ¡buenas noticias!</i>',
            estilos['BodyText'],
        ))
    else:
        # Estilo para citar reseñas: itálica, sangría, color gris.
        estilo_cita = ParagraphStyle(
            'CitaReseña',
            parent=estilos['BodyText'],
            fontSize=9,
            leftIndent=20,
            textColor=COLOR_GRIS_OSCURO,
            spaceAfter=8,
            italic=True,
        )

        for idx, fila in df_neg.iterrows():
            # Truncamos a 250 chars para que el reporte no se vuelva infinito.
            texto = fila['texto_original'][:250]
            if len(fila['texto_original']) > 250:
                texto += '...'

            # Agregamos la sucursal y confianza al final como metadata útil.
            sucursal_str = f" · {fila['sucursal']}" if 'sucursal' in fila else ""
            confianza_str = f" · {fila['confianza']*100:.0f}% confianza"
            story.append(Paragraph(
                f'"{texto}"<br/><font size=8 color="grey">{sucursal_str}{confianza_str}</font>',
                estilo_cita,
            ))

    # ------------------------------------------------------------------
    # SECCIÓN 5: RECOMENDACIONES AUTOMÁTICAS
    # ------------------------------------------------------------------
    story.append(Paragraph('Recomendaciones', estilo_seccion))

    # Aplicamos reglas simples de negocio para generar bullets accionables.
    # Esto NO es ML — son heurísticas que cualquier dueño de restaurante valida.
    recomendaciones = []

    if pct_neg > 30:
        recomendaciones.append(
            f"<b>Alerta: {pct_neg:.1f}% de reseñas son negativas</b> — significativamente "
            "alto. Recomendamos auditoría operativa urgente (calidad, servicio, tiempos)."
        )
    elif pct_neg > 15:
        recomendaciones.append(
            f"El {pct_neg:.1f}% de negativas requiere monitoreo. Identifica los patrones "
            "más frecuentes en la pestaña de Insights."
        )

    # Si tenemos sucursales, alertamos sobre las que tienen más negativas que el promedio.
    if 'sucursal' in df.columns:
        pct_neg_por_sucursal = (
            df[df['sentimiento'] == 'negativo'].groupby('sucursal').size()
            / df.groupby('sucursal').size() * 100
        ).fillna(0)
        # Sucursales con %neg > promedio + 10 puntos = outliers que requieren atención.
        umbral_alerta = pct_neg + 10
        peores = pct_neg_por_sucursal[pct_neg_por_sucursal > umbral_alerta]
        if len(peores) > 0:
            sucursales_str = ', '.join([f"<b>{s}</b> ({v:.0f}%)" for s, v in peores.items()])
            recomendaciones.append(
                f"Sucursales con desempeño por debajo del promedio: {sucursales_str}. "
                "Evaluar capacitación o refuerzo operativo en estos locales."
            )

    if confianza_prom < 70:
        recomendaciones.append(
            "Muchas reseñas tienen baja confianza del modelo. Considera reentrenar con "
            "datos del dominio específico o ajustar el umbral de revisión humana."
        )

    if not recomendaciones:
        recomendaciones.append(
            "Los indicadores están dentro de rangos saludables. Mantener el monitoreo "
            "regular y ejecutar este análisis mensualmente."
        )

    # Renderizamos las recomendaciones como bullets HTML (reportlab interpreta tags).
    for rec in recomendaciones:
        story.append(Paragraph(f'• {rec}', estilos['BodyText']))
        story.append(Spacer(1, 6))

    # ------------------------------------------------------------------
    # CONSTRUIR el PDF: doc.build() ejecuta el flow y escribe al buffer.
    # ------------------------------------------------------------------
    doc.build(story)

    # Volvemos al inicio del buffer y leemos todos los bytes para retornar.
    buffer.seek(0)
    return buffer.getvalue()
