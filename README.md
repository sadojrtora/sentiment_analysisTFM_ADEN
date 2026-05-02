---
title: Sentiment Pulse
emoji: 📊
colorFrom: green
colorTo: purple
sdk: docker
tags:
- streamlit
app_port: 8501
# sdk_version: 1.35.0
# app_file: app.py
pinned: false
license: mit
---

# 📊 Sentiment Pulse

**Plataforma de análisis de sentimientos para reseñas de restaurantes**

App web (Streamlit) que envuelve un modelo DistilBERT fine-tuneado sobre Yelp Reviews.
Acepta reseñas individuales o archivos CSV completos y genera un dashboard accionable
con KPIs, gráficas interactivas, alertas por sucursal, palabras clave en quejas y reportes PDF.

---

## 🗂️ Estructura del proyecto

```
sentiment_app/
├── app.py                      # Entry point Streamlit
├── pipeline.py                 # PipelineSentimientos (modelo + inferencia)
├── components/
│   ├── single_view.py          # Tab 1: Comentario único
│   ├── batch_view.py           # Tab 2: Análisis batch de CSV
│   └── insights_view.py        # Tab 3: Insights accionables
├── utils/
│   ├── viz.py                  # Funciones Plotly reutilizables
│   └── report.py               # Generador de PDF ejecutivo
├── modelo_sentimientos/        # ⚠️ TÚ debes colocar aquí el modelo entrenado
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── requirements.txt
└── README.md
```

---

## 🚀 Setup local (3 pasos)

### 1. Clonar / descargar este proyecto y posicionarse en su carpeta

```bash
cd sentiment_app
```

### 2. Crear entorno virtual e instalar dependencias

```bash
# Crear venv
python -m venv venv

# Activar venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Colocar el modelo entrenado

Copia tu modelo entrenado (el resultado de `Trainer.save_model()` del notebook)
en una carpeta llamada `modelo_sentimientos/` dentro del proyecto.

Si tu modelo está en Google Drive (caso típico tras entrenar en Colab):

```bash
# Opción A: descargarlo manualmente y copiarlo a ./modelo_sentimientos/
# Opción B: cambiar la ruta en el sidebar de la app cuando la abras
```

La app espera estos archivos dentro de la carpeta del modelo:
- `config.json`
- `model.safetensors` (o `pytorch_model.bin`)
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`

### 4. Lanzar la app

```bash
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`.

---

## 📋 Formato del CSV de entrada

La columna **obligatoria** es `texto`. Las demás son opcionales pero habilitan
visualizaciones extra:

| Columna     | Obligatoria | Función                                                 |
| ----------- | ----------- | ------------------------------------------------------- |
| `texto`     | ✅          | La reseña a analizar (cualquier idioma)                 |
| `sucursal`  | ❌          | Habilita gráfica de sentimiento por sucursal + alertas  |
| `fecha`     | ❌          | Habilita serie temporal (formato ISO: `YYYY-MM-DD`)     |
| `id`        | ❌          | Identificador único de cada reseña (se preserva en output) |

**Ejemplo:**

```csv
id,fecha,sucursal,texto
1,2024-11-01,Centro,"La hamburguesa estaba increíble!"
2,2024-11-02,Norte,"Cold burger and rude staff. Terrible."
3,2024-11-03,Sur,"Servicio normal, comida aceptable."
```

> **Tip:** la app incluye un botón "Descargar CSV de ejemplo" si subes la pestaña Batch sin archivo.

---

## 🎨 Las 3 tabs

### Tab 1 — Comentario único 🔍
Pegar una reseña → ver sentimiento, confianza, idioma detectado y barras de probabilidad por clase.
Incluye reseñas de ejemplo pre-cargadas para probar sin escribir.

### Tab 2 — Análisis batch (CSV) 📊
Subir CSV → procesamiento batch con barra de progreso → dashboard con:
- 4 KPI cards (total, % por sentimiento)
- Donut de distribución
- Histograma de confianza
- Barras apiladas por sucursal (si aplica)
- Serie temporal (si aplica)
- Tabla de reseñas para revisión humana (umbral ajustable)
- Descargas: CSV enriquecido + PDF ejecutivo

### Tab 3 — Insights accionables 💡
Análisis profundo sobre los resultados de la Tab 2:
- Top palabras / bigramas más frecuentes en reseñas negativas
- Alertas automáticas por sucursal (las que superan promedio + 10pp)
- Recomendaciones generadas por reglas de negocio

---

## 🚢 Deploy a Streamlit Community Cloud (gratis)

1. Sube todo el proyecto a un repo de GitHub (incluyendo el modelo o un script que lo descargue al iniciar).
2. Ve a [share.streamlit.io](https://share.streamlit.io) → "New app" → conecta tu repo.
3. Especifica `app.py` como entry point.
4. Deploy. En 2-3 minutos tienes URL pública.

> ⚠️ El modelo pesa ~265MB. Si tu repo excede el límite de GitHub (100MB por archivo),
> usa Git LFS o aloja el modelo en Hugging Face Hub y descárgalo en `cargar_pipeline()`.

---

## 🔧 Configuración avanzada

### Cambiar el umbral de "revisión humana"
En `components/batch_view.py`, ajusta `UMBRAL_CONFIANZA_DEFAULT`. También es ajustable
desde la UI con un slider.

### Cambiar los umbrales de alerta de negocio
En `components/insights_view.py`:
- `UMBRAL_NEG_ALERTA_GLOBAL`: % de negativas global que dispara alerta crítica.
- `UMBRAL_NEG_SUCURSAL_DELTA`: diferencia en pp sobre el promedio para alertar a una sucursal.

### Usar GPU
Si tu máquina tiene GPU NVIDIA, `pipeline.py` la detecta automáticamente con
`torch.cuda.is_available()`. No requiere configuración adicional.

---

## 🐛 Troubleshooting

**"No module named 'transformers'"**
→ `pip install -r requirements.txt` con el venv activado.

**"Error al cargar el modelo"**
→ Verifica que la ruta en el sidebar apunte a una carpeta con `config.json` y los demás archivos.

**El procesamiento de CSV es muy lento**
→ Si tienes GPU, sube el batch_size en "Configuración avanzada" a 64 o 128.
→ Si no, considera procesar el CSV en lotes más pequeños o usar Colab para el procesamiento.

**Google Translate falla con muchas reseñas**
→ Hay rate limits. Si tu CSV es 100% en español, considera traducirlo antes
   de subirlo (con DeepL o otra herramienta) y pasar todo en inglés.

---

## 📄 Licencia y créditos

TFM — Big Data & Business Analytics
ADEN International Business School
Donaldo Díaz · 2026

Modelo base: [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) (Apache 2.0)
Dataset: [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full)
