"""
================================================================================
 pipeline.py — Pipeline de inferencia de sentimientos para producción
================================================================================
 Este módulo encapsula TODO el flujo de procesamiento de una reseña:
   texto crudo  →  limpieza  →  detección de idioma  →  traducción (si aplica)
                →  tokenización  →  modelo DistilBERT  →  sentimiento + confianza

 Está extraído del notebook original (sentiment_analysis_restaurantes_v5.ipynb)
 y refactorizado con dos mejoras clave para uso en una app web:

   1. Método `predecir_batch()` que procesa N textos a la vez en GPU/CPU,
      usando DataLoader y batches. Esto es ~30x más rápido que llamar
      `predecir()` en un loop sobre cada fila de un DataFrame.

   2. Soporte para callback de progreso, para que Streamlit (u otra UI)
      pueda actualizar su barra de progreso mientras corre la inferencia.
================================================================================
"""

# --- Librerías estándar de Python ---
import re                                  # Expresiones regulares para limpieza de texto
from typing import Callable, List, Optional  # Type hints para mejorar la legibilidad y autocompletado

# --- Manejo numérico ---
import numpy as np                         # Operaciones vectorizadas sobre arrays (probabilidades, argmax, etc.)

# --- PyTorch: backend de inferencia ---
import torch                               # Framework de deep learning donde corre DistilBERT
import torch.nn.functional as F            # F.softmax para convertir logits en probabilidades

# --- Hugging Face Transformers ---
from transformers import (
    DistilBertTokenizerFast,                # Tokenizer rápido (implementado en Rust). Convierte texto → IDs numéricos
    DistilBertForSequenceClassification,    # Modelo DistilBERT con la cabeza de clasificación encima
)

# --- Detección y traducción de idioma ---
from langdetect import detect, LangDetectException  # detect(): retorna código ISO ('en', 'es'); excepción si falla
from langdetect import DetectorFactory                # Para fijar la semilla y hacerlo reproducible
from deep_translator import GoogleTranslator          # Wrapper sobre Google Translate; sin API key necesaria

# Fijamos seed de langdetect: por defecto langdetect es no-determinista (usa muestreo aleatorio)
# Sin esto, el mismo texto puede dar 'es' una vez y 'pt' otra. Con seed fijo es 100% reproducible.
DetectorFactory.seed = 777


# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

# Mapeo entero → nombre de clase. El modelo predice 0/1/2 y necesitamos texto humano.
# Este orden DEBE coincidir con el usado en el entrenamiento (ver CONFIG['LABEL_NAMES'] del notebook).
ETIQUETAS_SENTIMIENTO = {
    0: 'negativo',     # Reseñas de 1-2 estrellas Yelp
    1: 'neutral',      # Reseñas de 3 estrellas Yelp
    2: 'positivo',     # Reseñas de 4-5 estrellas Yelp
}

# Detectar el dispositivo donde corre el modelo. Si hay GPU NVIDIA disponible, la usamos
# (10-50x más rápido que CPU). Si no, fallback a CPU (más lento pero siempre funciona).
DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Longitud máxima de tokens que aceptará DistilBERT por reseña.
# DistilBERT-base soporta hasta 512 tokens, pero 256 es suficiente para reseñas típicas
# y ahorra ~50% memoria + 50% tiempo de inferencia. Un token ≈ 0.75 palabras en inglés.
LONGITUD_MAX_TOKENS = 256


# ============================================================================
# FUNCIONES DE PRE-PROCESAMIENTO
# ============================================================================

def limpiar_texto(texto: str, max_chars: int = 1000) -> Optional[str]:
    """
    Limpia y normaliza una reseña antes de pasarla al modelo.

    Pasos aplicados en orden (todos importantes para evitar ruido en el modelo):
        1. Bajar a minúsculas (DistilBERT-uncased fue entrenado así)
        2. Eliminar URLs (no aportan sentimiento)
        3. Eliminar menciones (@user) y hashtags (#topic)
        4. Eliminar caracteres especiales raros (emojis, símbolos)
        5. Normalizar espacios en blanco
        6. Truncar al máximo de caracteres
        7. Descartar textos < 10 chars (insuficientes)

    Args:
        texto: La reseña original tal como llegó del usuario.
        max_chars: Tope superior de caracteres antes del truncado (default 1000).

    Returns:
        El texto limpio listo para tokenizar, o None si quedó vacío/inválido.
    """
    # Validación defensiva: si no es string o está vacío, retornamos None.
    # `not texto.strip()` cubre tanto "" como "   " (solo espacios).
    if not isinstance(texto, str) or not texto.strip():
        return None

    # Paso 1: a minúsculas. DistilBERT 'uncased' espera todo en minúscula.
    # Si pasáramos "GREAT" lo procesaría como token desconocido, perdiendo señal.
    texto = texto.lower()

    # Paso 2: eliminar URLs. El patrón cubre tanto http(s):// como www.
    # \S+ = uno o más caracteres NO whitespace (toma todo hasta el siguiente espacio).
    texto = re.sub(r'http\S+|www\.\S+', '', texto)

    # Paso 3: menciones y hashtags. Ruido típico de redes sociales que no aporta sentimiento.
    # \w+ = una o más letras/dígitos/guiones bajos (o sea, el "nombre" después de @ o #).
    texto = re.sub(r'@\w+|#\w+', '', texto)

    # Paso 4: solo conservamos letras (incluye á-ú-ñ), números, y puntuación básica.
    # Cualquier otro símbolo (emoji 🍔, ★, ©) lo reemplazamos por espacio.
    # El flag re.UNICODE asegura que \w respete acentos en español.
    texto = re.sub(r"[^a-záéíóúüñ\w\s.,!?']", ' ', texto, flags=re.UNICODE)

    # Paso 5a: convertir saltos de línea, tabs y retornos a espacios simples.
    texto = re.sub(r'[\n\r\t]+', ' ', texto)

    # Paso 5b: colapsar múltiples espacios consecutivos en uno solo.
    # Después usamos .strip() para quitar espacios al inicio/final.
    texto = re.sub(r'\s{2,}', ' ', texto).strip()

    # Paso 6: si excede el límite, cortar en el último espacio para no partir palabras.
    # rsplit(' ', 1) divide desde la derecha; [0] toma la parte izquierda (sin la palabra cortada).
    if len(texto) > max_chars:
        texto = texto[:max_chars].rsplit(' ', 1)[0]

    # Paso 7: validación final. Textos < 10 chars probablemente son ruido ("ok", "wow").
    # Mejor descartarlos que darles una etiqueta incorrecta.
    if len(texto) < 10:
        return None

    return texto


def detectar_idioma(texto: str) -> str:
    """
    Detecta el idioma de un texto usando langdetect (basado en n-gramas estadísticos).

    Args:
        texto: Texto en cualquier idioma.

    Returns:
        Código ISO 639-1 de 2 letras ('en', 'es', 'fr', ...) o 'unknown' si falla.
    """
    try:
        # langdetect requiere ~20 caracteres mínimo para ser confiable.
        # Con menos, sus n-gramas no tienen suficiente evidencia y da resultados aleatorios.
        if len(texto.strip()) < 20:
            return 'unknown'
        # detect() es la API principal: retorna el idioma más probable como string ISO.
        return detect(texto)
    except LangDetectException:
        # Excepción específica de langdetect cuando no puede determinar el idioma
        # (texto muy ambiguo, mezcla de idiomas, etc.). No queremos romper el pipeline.
        return 'unknown'


def traducir_a_ingles(texto: str, idioma_origen: str) -> str:
    """
    Traduce un texto al inglés usando Google Translate (vía deep-translator).

    Estrategia "translate-then-classify": como el modelo fue entrenado solo en inglés,
    es más preciso traducir el texto a inglés que intentar clasificar en español
    con un modelo que apenas vio español.

    IMPORTANTE: este método requiere conexión a internet. Si falla (red, rate limit
    de Google, idioma raro), retornamos el texto ORIGINAL en lugar de fallar — así
    el pipeline sigue corriendo aunque la traducción no funcione (fail-safe).

    Args:
        texto: Texto en idioma extranjero a traducir.
        idioma_origen: Código ISO del idioma origen ('es', 'fr', etc.).

    Returns:
        Texto traducido al inglés, o el original si la traducción falló.
    """
    try:
        # Inicializamos el traductor con origen y destino. 'en' = inglés.
        traductor = GoogleTranslator(source=idioma_origen, target='en')
        # Google Translate tiene un límite de ~5000 caracteres por request.
        # Truncamos a 4500 por seguridad (margen para el overhead del request).
        texto_truncado = texto[:4500] if len(texto) > 4500 else texto
        # translate() hace el request HTTP a Google y retorna el texto traducido.
        return traductor.translate(texto_truncado)
    except Exception as e:
        # Capturamos CUALQUIER excepción (red, parsing, rate limit) para no romper el pipeline.
        # En producción esto debería loggearse en lugar de imprimirse.
        print(f"[traducir_a_ingles] Error: {e}. Usando texto original.")
        return texto


def preparar_texto_para_modelo(texto: str) -> dict:
    """
    Pipeline completo de preparación: limpieza → detección → traducción.

    Esta es la función que TODA reseña debe pasar antes del modelo.
    Retorna un dict con metadata útil (idioma detectado, si fue traducido)
    para que la UI pueda mostrar al usuario qué pasó con su texto.

    Args:
        texto: Reseña cruda del usuario.

    Returns:
        dict con campos:
          - texto_original: el texto tal como llegó
          - texto_limpio:   después de limpieza
          - idioma:         código ISO detectado (o 'unknown')
          - texto_modelo:   versión final que va al modelo (en inglés)
          - traducido:      bool, True si hubo traducción ES→EN
    """
    # Estructura de retorno con valores por defecto.
    # Iremos llenando los campos según avance el pipeline.
    resultado = {
        'texto_original': texto,
        'texto_limpio': None,
        'idioma': 'unknown',
        'texto_modelo': None,
        'traducido': False,
    }

    # Paso 1: limpieza. Si retorna None (texto inválido), abortamos temprano.
    texto_limpio = limpiar_texto(texto)
    if texto_limpio is None:
        return resultado  # texto_modelo queda en None → la UI sabrá que no procesó

    resultado['texto_limpio'] = texto_limpio

    # Paso 2: detectar idioma sobre el texto YA limpio (más confiable que sobre el crudo).
    idioma = detectar_idioma(texto_limpio)
    resultado['idioma'] = idioma

    # Paso 3: si no es inglés, traducir. Si ya es inglés, pasarlo directo al modelo.
    if idioma != 'en' and idioma != 'unknown':
        # Solo traducimos cuando estamos seguros del idioma origen.
        # Si es 'unknown', mejor mandar el texto tal cual (puede ser inglés mal detectado).
        resultado['texto_modelo'] = traducir_a_ingles(texto_limpio, idioma)
        resultado['traducido'] = True
    else:
        resultado['texto_modelo'] = texto_limpio
        resultado['traducido'] = False

    return resultado


# ============================================================================
# CLASE PRINCIPAL DEL PIPELINE
# ============================================================================

class PipelineSentimientos:
    """
    Pipeline production-ready para clasificación de sentimientos en reseñas.

    Encapsula:
      - Carga del modelo y tokenizer (una sola vez al instanciar)
      - Predicción individual: `predecir(texto)`
      - Predicción por lotes (vectorizada): `predecir_batch(textos)`

    Ejemplo de uso:
        pipeline = PipelineSentimientos('./modelo_sentimientos')
        # Caso 1: una reseña
        r = pipeline.predecir("La comida estaba deliciosa!")
        # Caso 2: muchas reseñas (mucho más eficiente que un loop)
        resultados = pipeline.predecir_batch(["Texto 1", "Texto 2", ...])
    """

    def __init__(self, ruta_modelo: str):
        """
        Carga el modelo entrenado desde disco al dispositivo (GPU/CPU).

        Esto es la operación CARA (~5-10 segundos). Por eso lo hacemos una sola vez
        al instanciar la clase, no en cada predicción. En Streamlit esto se cachea
        con @st.cache_resource para que no se recargue entre interacciones.

        Args:
            ruta_modelo: Carpeta con los archivos del modelo (config.json, model.safetensors,
                         tokenizer.json, vocab.txt, etc.). Es lo que produce el `Trainer.save_model()`
                         del notebook original.
        """
        REPO_HF = "sadojrtora/distilbert-sentimientosTFM"  # tu repo en HF

        # Cargamos el tokenizer: convierte texto → tokens → IDs numéricos.
        # `Fast` significa implementación en Rust (~10x más rápido que la versión Python pura).
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(REPO_HF)
       
        #self.tokenizer = DistilBertTokenizerFast.from_pretrained(ruta_modelo)  esta es la version local, la de arriba es la version del repo en Hugging Face. Ambas funcionan igual, solo cambia la fuente de donde se cargan los archivos del modelo/tokenizer.

        # Cargamos el modelo: arquitectura DistilBERT + cabeza de clasificación con 3 outputs.
        # Esto carga los pesos fine-tuneados que entrenamos sobre Yelp Reviews.
        self.modelo    = DistilBertForSequenceClassification.from_pretrained(REPO_HF)

        # self.modelo = DistilBertForSequenceClassification.from_pretrained(ruta_modelo) esta es la version local, la de arriba es la version del repo en Hugging Face. Ambas funcionan igual, solo cambia la fuente de donde se cargan los archivos del modelo/tokenizer.

        # Movemos el modelo al dispositivo. Si hay GPU disponible (DISPOSITIVO='cuda'),
        # la inferencia será mucho más rápida. Si no, corre en CPU.
        self.modelo.to(DISPOSITIVO)

        # Modo eval(): desactiva dropout y batch normalization. Crítico durante inferencia
        # porque queremos predicciones DETERMINISTAS (mismo input → mismo output siempre).
        self.modelo.eval()

        # Guardamos el mapeo entero → etiqueta para reutilizarlo en cada predicción.
        self.id2label = ETIQUETAS_SENTIMIENTO

    # ------------------------------------------------------------------------
    # PREDICCIÓN INDIVIDUAL
    # ------------------------------------------------------------------------

    def predecir(self, texto: str) -> dict:
        """
        Predice el sentimiento de UNA reseña. Útil para análisis interactivo.

        Para procesar muchas reseñas, usar `predecir_batch()` que es ~30x más rápido.

        Returns:
            dict con: texto_original, sentimiento, confianza, probabilidades (dict),
                      idioma_detectado, traducido. O dict con 'error' si falló.
        """
        # Paso 1: pre-procesamiento (limpieza + idioma + traducción).
        prep = preparar_texto_para_modelo(texto)

        # Si el texto era inválido (muy corto, vacío), retornamos error explicativo
        # para que la UI pueda mostrar un mensaje útil al usuario.
        if not prep['texto_modelo']:
            return {
                'error': 'El texto es demasiado corto o no pudo ser procesado.',
                'texto_original': texto,
            }

        # Paso 2: tokenizar el texto. Esto convierte string → tensores que entiende el modelo.
        # `return_tensors='pt'` retorna tensores PyTorch (vs numpy o tensorflow).
        # `truncation=True` corta si excede max_length (DistilBERT no acepta más de 512 tokens).
        # `padding=True` rellena con [PAD] hasta una longitud fija (necesario para batches).
        tokens = self.tokenizer(
            prep['texto_modelo'],
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=LONGITUD_MAX_TOKENS,
        )

        # Movemos los tensores al mismo dispositivo donde está el modelo (GPU o CPU).
        # Si los tensores quedan en CPU y el modelo en GPU, PyTorch lanza un error.
        tokens = {k: v.to(DISPOSITIVO) for k, v in tokens.items()}

        # Paso 3: inferencia. `torch.no_grad()` desactiva el cálculo de gradientes,
        # ahorrando memoria y tiempo (no estamos entrenando, solo prediciendo).
        with torch.no_grad():
            # El modelo retorna un objeto con `.logits` (scores brutos sin normalizar).
            # Para 3 clases, logits es de forma (1, 3) — un score por clase.
            outputs = self.modelo(**tokens)
            logits = outputs.logits

        # Paso 4: convertir logits → probabilidades. softmax garantiza que sumen 1.0
        # y que estén en [0, 1]. dim=-1 aplica softmax sobre la dimensión de clases.
        # .cpu() trae el tensor de GPU a CPU para poder convertirlo a numpy.
        # .numpy()[0] extrae el primer (y único) ejemplo del batch.
        probabilidades = F.softmax(logits, dim=-1).cpu().numpy()[0]

        # argmax = índice de la probabilidad máxima → clase predicha (0, 1 o 2).
        # int() porque numpy retorna np.int64 y queremos int Python estándar (mejor para JSON).
        idx_predicho = int(np.argmax(probabilidades))

        # Construimos el dict de respuesta. La UI usará estos campos directamente.
        return {
            'texto_original': prep['texto_original'],
            'texto_limpio': prep['texto_limpio'],
            'idioma_detectado': prep['idioma'],
            'traducido': prep['traducido'],
            'texto_modelo': prep['texto_modelo'],  # útil para debugging
            'sentimiento': self.id2label[idx_predicho],
            'confianza': float(probabilidades[idx_predicho]),  # float() para serializar a JSON
            # Dict con probabilidad por clase, útil para gráficas de barras.
            'probabilidades': {
                self.id2label[i]: float(p) for i, p in enumerate(probabilidades)
            },
        }

    # ------------------------------------------------------------------------
    # PREDICCIÓN POR LOTES (vectorizada — el truco de velocidad)
    # ------------------------------------------------------------------------

    def predecir_batch(
        self,
        textos: List[str],
        batch_size: int = 32,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[dict]:
        """
        Procesa una lista de textos de forma EFICIENTE usando inferencia por lotes.

        ¿Por qué es más rápido que llamar `predecir()` en un loop?
        - GPU: hace operaciones matriciales en paralelo. Procesar 32 textos a la vez
          tarda casi lo mismo que procesar 1 (la GPU está infrautilizada con 1).
        - CPU: aprovecha vectorización SIMD y reduce el overhead de Python.
        - El tokenizer `Fast` también procesa listas mucho más rápido que strings sueltos.

        Resultado típico en GPU: 32 textos en ~0.05s vs 32 textos en loop: ~3.0s.

        Args:
            textos: Lista de reseñas crudas a clasificar.
            batch_size: Cuántos textos procesar por iteración. 32 es buen default.
                        Subir a 64-128 si tienes GPU con mucha VRAM. Bajar si OOM.
            callback: Función opcional `callback(procesados, total)` que se llama
                      después de cada batch. Útil para barras de progreso en Streamlit.

        Returns:
            Lista de dicts (mismo formato que `predecir()`), uno por cada texto de entrada,
            EN EL MISMO ORDEN. Si un texto era inválido, su dict tendrá 'error'.
        """
        # ------- FASE 1: pre-procesamiento de TODOS los textos -------
        # Limpiamos y traducimos cada texto. Esta fase es secuencial (no se vectoriza fácil
        # porque incluye llamadas HTTP a Google Translate). Pero es la parte rápida.
        preparados = []
        for texto in textos:
            preparados.append(preparar_texto_para_modelo(texto))

        # Separamos los textos VÁLIDOS (que pasarán al modelo) de los INVÁLIDOS.
        # Guardamos los índices originales para poder reconstruir el orden al final.
        indices_validos = []          # posiciones en la lista original que tienen texto utilizable
        textos_para_modelo = []       # los textos limpios listos para tokenizar
        for i, prep in enumerate(preparados):
            if prep['texto_modelo']:  # texto_modelo es None si fue rechazado en limpieza
                indices_validos.append(i)
                textos_para_modelo.append(prep['texto_modelo'])

        # ------- FASE 2: inferencia por lotes -------
        # Acumulamos las probabilidades de cada batch en una lista para concatenar al final.
        todas_las_probs = []  # cada elemento será un array de forma (batch_size, 3)
        total_validos = len(textos_para_modelo)

        # `torch.no_grad()` envuelve TODO el loop para no construir grafo de gradientes
        # (estamos prediciendo, no entrenando). Esto reduce uso de RAM significativamente.
        with torch.no_grad():
            # Recorremos los textos de batch_size en batch_size.
            # range(0, N, step) genera 0, step, 2*step, ... hasta N exclusive.
            for inicio in range(0, total_validos, batch_size):
                fin = min(inicio + batch_size, total_validos)  # cuidado en el último batch
                batch_textos = textos_para_modelo[inicio:fin]

                # Tokenizar TODO el batch a la vez. El tokenizer aplica padding automáticamente
                # para que todos los textos del batch tengan la misma longitud (requisito de tensores).
                # Internamente esto usa el tokenizer Rust, MUCHO más rápido que loop Python.
                tokens = self.tokenizer(
                    batch_textos,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=LONGITUD_MAX_TOKENS,
                )
                # Mover al dispositivo (GPU/CPU).
                tokens = {k: v.to(DISPOSITIVO) for k, v in tokens.items()}

                # Forward pass del modelo. logits tiene forma (batch_size, 3).
                outputs = self.modelo(**tokens)
                # Softmax → probabilidades. .cpu().numpy() trae los datos de vuelta a CPU/numpy
                # para procesamiento posterior fuera del grafo PyTorch.
                probs_batch = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                todas_las_probs.append(probs_batch)

                # Notificar progreso a la UI (si hay callback). Este es el hook clave para Streamlit.
                # Pasamos cuántos textos procesamos hasta ahora y el total, para calcular el %.
                if callback is not None:
                    callback(fin, total_validos)

        # Concatenar todos los batches en un solo array de forma (total_validos, 3).
        # Si todas_las_probs está vacío (caso edge: todos los textos eran inválidos),
        # creamos un array vacío para no romper el código que sigue.
        if todas_las_probs:
            probs_completas = np.concatenate(todas_las_probs, axis=0)
        else:
            probs_completas = np.zeros((0, 3))  # array vacío con forma compatible

        # ------- FASE 3: reconstruir resultados en el ORDEN ORIGINAL -------
        # Recorremos la lista de preparados original y vamos pegando los resultados de modelo
        # solo donde el texto era válido. Los inválidos reciben un dict de error.
        resultados = []
        idx_modelo = 0  # cursor sobre probs_completas (incrementa solo en válidos)

        for i, prep in enumerate(preparados):
            # Caso A: texto válido → tomamos su predicción del array de probabilidades.
            if i in set(indices_validos):  # nota: convertir a set una sola vez sería más eficiente
                probs = probs_completas[idx_modelo]
                clase_predicha = int(np.argmax(probs))

                resultados.append({
                    'texto_original': prep['texto_original'],
                    'texto_limpio': prep['texto_limpio'],
                    'idioma_detectado': prep['idioma'],
                    'traducido': prep['traducido'],
                    'sentimiento': self.id2label[clase_predicha],
                    'confianza': float(probs[clase_predicha]),
                    # Probabilidades planas (más fácil de pasar a un DataFrame que un dict anidado).
                    'prob_negativo': float(probs[0]),
                    'prob_neutral': float(probs[1]),
                    'prob_positivo': float(probs[2]),
                })
                idx_modelo += 1

            # Caso B: texto inválido → registramos el error pero mantenemos la fila
            # para que la lista de salida tenga el mismo largo que la entrada.
            else:
                resultados.append({
                    'texto_original': prep['texto_original'],
                    'texto_limpio': None,
                    'idioma_detectado': prep['idioma'],
                    'traducido': False,
                    'sentimiento': 'error',
                    'confianza': 0.0,
                    'prob_negativo': 0.0,
                    'prob_neutral': 0.0,
                    'prob_positivo': 0.0,
                    'error': 'Texto demasiado corto o vacío',
                })

        return resultados
