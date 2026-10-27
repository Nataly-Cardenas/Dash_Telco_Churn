"""Asistente del dashboard: responde preguntas sobre el proyecto y sobre cómo usar el dash.

Funciona en dos modos:
- Con la variable de entorno ANTHROPIC_API_KEY, redacta la respuesta con Claude a partir del contexto recuperado.
- Sin ella (o si la API falla), responde con la entrada más parecida de la base de conocimiento local.
"""
import os
import re
from functools import lru_cache
import numpy as np
from dash import dcc, html, Input, Output, State, ALL, ctx, no_update
from sklearn.feature_extraction.text import TfidfVectorizer
import analytics as a
from i18n import T, N, set_lang, get_lang
from icons import icon

MAX_QUESTIONS = 20
MODEL = 'claude-opus-5-5'
BOOK = 'https://nataly-cardenas.github.io/Telco_Churn_Entregable/'

# ---------------------------------------------------------------- base de conocimiento
# Cada entrada: (preguntas de ejemplo, respuesta en español, respuesta en inglés).
FAQ = [
    (['qué es el churn', 'qué es el abandono de clientes', 'what is churn'],
     'El *churn* o abandono es la cancelación del servicio por parte de un cliente. En este proyecto la variable `Churn` indica si el cliente se fue durante el último mes observado.',
     'Churn is when a customer cancels the service. In this project the `Churn` column says whether the customer left during the last observed month.'),
    (['de qué trata el proyecto', 'cuál es el objetivo', 'what is the project about', 'goal of the project'],
     'El proyecto busca identificar a los clientes de una empresa de telecomunicaciones con mayor riesgo de abandonar el servicio y los factores asociados, comparando de forma reproducible 112 configuraciones de modelos de clasificación.',
     'The project aims to identify the telecom customers most likely to cancel and the factors associated with it, by reproducibly comparing 112 classification model configurations.'),
    (['qué datos se usaron', 'cuántos clientes hay en el dataset', 'de dónde salen los datos', 'cuántas variables hay', 'which data was used', 'how many customers are in the dataset', 'where does the data come from'],
     'Se usó el conjunto público **Telco Customer Churn** de IBM (Kaggle): 7,043 clientes y 21 columnas con servicios contratados, datos de la cuenta y perfil demográfico. Se dividió 80/20: 5,634 clientes de entrenamiento y 1,409 de test.',
     'The public IBM **Telco Customer Churn** dataset (Kaggle): 7,043 customers and 21 columns with services, account data and demographics. It was split 80/20: 5,634 training and 1,409 test customers.'),
    (['cuál es la tasa de abandono', 'cuántos clientes abandonan', 'qué porcentaje se va', 'what is the churn rate', 'how many customers churn', 'what percentage leaves'],
     'Abandonan 1,869 de 7,043 clientes, es decir **26.5 %**. Por eso es un problema desbalanceado: la clase de interés es minoritaria.',
     '1,869 of 7,043 customers churn, i.e. **26.5 %**. The problem is imbalanced: the class of interest is the minority.'),
    (['qué clientes abandonan más', 'quiénes se van más', 'cuáles son los factores de riesgo', 'qué variables influyen en el abandono', 'por qué se van los clientes', 'which customers churn the most', 'who leaves the most', 'what are the risk factors', 'which variables matter', 'why do customers leave'],
     'Las tasas más altas aparecen con **cheque electrónico (45.7 %)**, **contrato mes a mes (42.7 %)**, **fibra óptica (42.1 %)**, sin seguridad en línea (41.9 %) y sin soporte técnico (41.8 %). SHAP confirma que **contrato, servicio de internet y antigüedad** son las variables que más pesan en los modelos.',
     'The highest rates appear with **electronic check (45.7 %)**, **month-to-month contracts (42.7 %)**, **fiber optic (42.1 %)**, no online security (41.9 %) and no tech support (41.8 %). SHAP confirms that **contract, internet service and tenure** weigh the most in the models.'),
    (['cuál fue el mejor modelo', 'qué modelo ganó', 'qué modelo funcionó mejor', 'cuál es el modelo con mejor f1', 'which was the best model', 'which model won', 'what is the best model'],
     '**Random Forest con SMOTE** obtuvo el mayor F1 medio (0.638 ± 0.023), empatado con **XGBoost con SMOTE** (0.638) y seguido por la regresión logística (0.634). Las diferencias no son estadísticamente detectables, así que no hay un ganador demostrado. Míralo en *Arena de modelos*.',
     '**Random Forest with SMOTE** had the highest mean F1 (0.638 ± 0.023), tied with **XGBoost with SMOTE** (0.638) and followed by logistic regression (0.634). The differences are not statistically detectable, so there is no proven winner. See *Model arena*.'),
    (['qué es el f1', 'qué métrica se usó', 'por qué f1', 'what is f1', 'which metric'],
     'El **F1** de la clase de abandono es la media armónica de Precision y Recall: F1 = 2TP / (2TP + FP + FN). Se eligió porque equilibra detectar a quienes se van y no generar demasiadas falsas alertas; con clases desbalanceadas es más informativo que la exactitud.',
     'The **F1** of the churn class is the harmonic mean of precision and recall: F1 = 2TP / (2TP + FP + FN). It balances catching churners and avoiding false alarms, and is more informative than accuracy with imbalanced classes.'),
    (['qué es el auc', 'auc roc', 'what is auc'],
     'El **AUC-ROC** mide qué tan bien el modelo ordena a los clientes por riesgo: es la probabilidad de que un cliente que abandona reciba más puntaje que uno que se queda. Los finalistas tienen AUC ≈ 0.84 en el test.',
     '**AUC-ROC** measures how well the model ranks customers by risk: the probability that a churner gets a higher score than a non-churner. The finalists reach AUC ≈ 0.84 on the test set.'),
    (['qué es smote y por qué se usó', 'cómo se trató el desbalance', 'qué técnicas de balanceo se usaron', 'what is smote and why was it used', 'how was class imbalance handled'],
     '**SMOTE** crea clientes sintéticos de la clase minoritaria interpolando entre vecinos cercanos. Se comparó con ADASYN, pesos de clase y no balancear; SMOTE mejoró el F1 en 6 de las 7 familias. Siempre se aplica solo sobre los datos de entrenamiento.',
     '**SMOTE** creates synthetic minority-class customers by interpolating between near neighbors. It was compared with ADASYN, class weights and no balancing; SMOTE improved F1 in 6 of the 7 families. It is only ever applied to training data.'),
    (['qué es la validación cruzada anidada', 'cómo se validaron los modelos', 'nested cross validation'],
     'En la **validación cruzada anidada** un bucle interno (3 folds) elige los hiperparámetros y uno externo (5 folds) mide el desempeño con datos que no participaron en esa elección. Así se evita una estimación optimista.',
     'In **nested cross-validation** an inner loop (3 folds) picks the hyperparameters and an outer loop (5 folds) measures performance on data that played no part in that choice, avoiding optimistic estimates.'),
    (['qué es la fuga de información', 'data leakage'],
     'La **fuga de información** ocurre cuando datos de validación o test influyen en el entrenamiento. Para evitarla, la imputación, la codificación, el escalado y el balanceo se ajustan solo con el entrenamiento de cada partición, dentro de un pipeline.',
     '**Data leakage** happens when validation or test data influence training. To avoid it, imputation, encoding, scaling and balancing are fitted only on each split\'s training data, inside a pipeline.'),
    (['cuántas configuraciones se probaron', 'cuántos modelos se probaron', 'qué modelos se compararon', 'how many models were tested', 'which models were compared', 'how many configurations'],
     'Se compararon **112 configuraciones**: 7 familias (regresión logística, KNN, Naive Bayes, árbol de decisión, Random Forest, XGBoost y SVM) × 4 balanceos (ninguno, pesos, SMOTE, ADASYN) × 4 buscadores de hiperparámetros (Grid, Random, TPE y genético).',
     '**112 configurations** were compared: 7 families (logistic regression, KNN, Naive Bayes, decision tree, Random Forest, XGBoost and SVM) × 4 balancing strategies (none, weights, SMOTE, ADASYN) × 4 hyperparameter searches (Grid, Random, TPE and genetic).'),
    (['qué optimizador es mejor', 'grid random tpe genético', 'which optimizer'],
     'Con el mismo presupuesto de 24 evaluaciones, TPE, Random y el algoritmo genético solo mejoraron a Grid con presupuestos bajos; al final quedaron prácticamente empatados. El balanceo influyó mucho más que el buscador. Mira la *Carrera de buscadores*.',
     'With the same budget of 24 evaluations, TPE, Random and the genetic algorithm only beat Grid at low budgets; in the end they were practically tied. Balancing mattered far more than the search method. See the *Search race*.'),
    (['qué es la calibración', 'brier ece', 'what is calibration'],
     'La **calibración sigmoide** (Platt) ajusta las probabilidades para que reflejen la frecuencia real. Redujo a la vez el puntaje de Brier y el ECE en 80 de 96 configuraciones.',
     '**Sigmoid calibration** (Platt) adjusts probabilities so they match observed frequencies. It reduced both the Brier score and ECE in 80 of 96 configurations.'),
    (['hay diferencias significativas', 'pruebas estadísticas', 'delong friedman', 'statistical tests'],
     'Friedman detecta diferencias entre las 112 configuraciones, pero Nemenyi no separa a las mejores. DeLong con corrección de Holm da p = 1.0 entre los tres finalistas y los intervalos bootstrap del AUC incluyen el cero.',
     'Friedman detects differences among the 112 configurations, but Nemenyi does not separate the best ones. DeLong with Holm correction gives p = 1.0 among the three finalists and the bootstrap AUC intervals include zero.'),
    (['qué es shap', 'qué es lime', 'interpretabilidad', 'what is shap'],
     '**SHAP** reparte cada predicción entre las variables (valores de Shapley) y su promedio da la importancia global. **LIME** explica un cliente concreto con un modelo simple local. Describen al modelo, no causas del abandono. Ver *¿Por qué se van?*.',
     '**SHAP** splits each prediction among the variables (Shapley values) and its average gives global importance. **LIME** explains one customer with a simple local model. They describe the model, not causes of churn. See *Why do they leave?*.'),
    (['cómo uso el simulador', 'perfil de riesgo', 'how do i use the simulator'],
     'En *Simulador → Perfil de riesgo* eliges contrato, internet, método de pago, antigüedad y cargo mensual; el dash busca clientes de entrenamiento parecidos y muestra su tasa de abandono observada. En *Clientes reales* ves las probabilidades de los 3 finalistas para un cliente del test, y en *Umbral de alerta* mueves la alarma de retención.',
     'In *Simulator → Risk profile* you choose contract, internet, payment method, tenure and monthly charge; the dash finds similar training customers and shows their observed churn rate. *Real customers* shows the 3 finalists\' probabilities for a test customer, and *Alert threshold* lets you move the retention alarm.'),
    (['qué pasa si cambio el umbral', 'umbral de alerta', 'threshold'],
     'Bajar el umbral detecta más clientes que se van (más Recall) pero genera más falsas alertas (menos Precision). Elegirlo bien exige conocer el costo de contactar a un cliente y el de perderlo.',
     'Lowering the threshold catches more churners (higher recall) but raises more false alarms (lower precision). Choosing it well requires knowing the cost of contacting a customer and of losing one.'),
    (['cómo descargo los datos', 'cómo descargo el csv', 'puedo exportar la tabla', 'how do i download the data', 'download csv', 'export the table'],
     'Ve a *Datos → Explorador*, elige las columnas y pulsa **Descargar CSV**. Se descarga un solo archivo con la selección de los filtros superiores.',
     'Go to *Data → Explorer*, pick the columns and press **Download CSV**. You get a single file with the current selection of the top filters.'),
    (['cómo cambio el idioma', 'cómo cambio a modo oscuro', 'cómo pongo el modo claro', 'how do i change the language', 'how do i switch to dark mode', 'light mode'],
     'Arriba a la derecha están los botones **ES / EN** para el idioma y el botón de sol/luna para cambiar entre modo claro y oscuro. La elección se recuerda en tu navegador.',
     'At the top right you\'ll find the **ES / EN** buttons for the language and the sun/moon button to switch between light and dark mode. Your choice is remembered in your browser.'),
    (['qué secciones tiene el dash', 'cómo navego', 'what sections'],
     'El dash tiene 8 secciones: El problema, Datos, Preparación, Arena de modelos, Simulador, ¿Por qué se van?, Conclusiones y Autoras. La portada es interactiva: pasa el cursor sobre los puntos rojos para retener clientes.',
     'The dash has 8 sections: The problem, Data, Preparation, Model arena, Simulator, Why do they leave?, Conclusions and Authors. The cover is interactive: hover over the red dots to retain customers.'),
    (['quiénes son las autoras', 'quién hizo el proyecto', 'who are the authors'],
     'Las autoras son **Laura S. Rivera Figueredo** y **Natalý A. Cárdenas Izaquita**, del Departamento de Matemáticas, Física y Ciencia de Datos de la Universidad del Norte (Barranquilla, Colombia).',
     'The authors are **Laura S. Rivera Figueredo** and **Natalý A. Cárdenas Izaquita**, from the Department of Mathematics, Physics and Data Science at Universidad del Norte (Barranquilla, Colombia).'),
    (['dónde está el jupyter book', 'dónde leo el paper', 'jupyter book', 'paper'],
     f'Todo el proyecto está en el [Jupyter Book]({BOOK}), incluido el [paper]({BOOK}paper.html). También hay un botón en la barra superior.',
     f'The whole project is in the [Jupyter Book]({BOOK}), including the [paper]({BOOK}paper.html). There is also a button in the top bar.'),
    (['cuáles son las limitaciones', 'limitations'],
     'Se usó un único conjunto de datos, el test ya había sido expuesto (es evidencia suplementaria), los datos son una foto de un momento y no hay costos de intervención, así que el F1 no se traduce directamente en dinero.',
     'A single dataset was used, the test set had already been seen (it is supplementary evidence), the data are a snapshot in time and there are no intervention costs, so F1 does not translate directly into money.'),
    (['cómo se preparó la información', 'preprocesamiento', 'valores faltantes', 'preprocessing', 'missing values'],
     'TotalCharges tenía 11 vacíos, todos de clientes con antigüedad 0, que se reemplazaron por 0. Se unificaron las categorías "sin internet/sin teléfono" en "No", se creó el cargo promedio mensual, se codificó con One-Hot y se estandarizó. Todo dentro del pipeline, sin fuga de información.',
     'TotalCharges had 11 blanks, all for customers with tenure 0, replaced by 0. "No internet/no phone" categories were merged into "No", an average monthly charge was created, and data were one-hot encoded and standardized, all inside the pipeline with no leakage.'),
    (['qué sigue', 'trabajo futuro', 'future work', 'next steps'],
     'Los próximos pasos son validar en clientes futuros (validación temporal), incorporar costos explícitos para elegir el umbral y diseñar un modelo propio que supere a las referencias en desempeño y tiempo.',
     'Next steps are validating on future customers (temporal validation), adding explicit costs to choose the threshold, and designing a proposed model that beats the baselines in performance and time.'),
]

FAQ.append((['qué significa el color del mapa de riesgo', 'cómo leo el sunburst', 'qué muestra el mapa de riesgo', 'how do i read the risk map', 'what does the sunburst color mean'],
            'En *Datos → Segmentos de riesgo*, el mapa circular va de contrato (centro) a internet y método de pago (afuera). El tamaño de cada porción es el número de clientes y el color, la tasa de abandono: coral es más riesgo, turquesa menos. Haz clic en una porción para profundizar.',
            'In *Data → Risk segments*, the circular map goes from contract (center) to internet and payment method (outside). Slice size is the number of customers and color is the churn rate: coral means more risk, teal less. Click a slice to drill down.'))
FAQ.append((['qué variables tiene el dataset', 'qué significa tenure', 'qué es monthlycharges', 'what variables are there', 'what does tenure mean'],
            'Hay 19 variables candidatas: de **cuenta** (contrato, método de pago, factura electrónica, `tenure` = antigüedad en meses, `MonthlyCharges` = cargo mensual, `TotalCharges` = cargo acumulado), de **servicios** (teléfono, líneas múltiples, internet, seguridad y respaldo en línea, protección de dispositivos, soporte técnico, streaming) y **demográficas** (género, adulto mayor, pareja, dependientes). Puedes explorarlas en *El problema* y en *Datos*.',
            'There are 19 candidate variables: **account** (contract, payment method, paperless billing, `tenure` = months as a customer, `MonthlyCharges`, `TotalCharges`), **services** (phone, multiple lines, internet, online security and backup, device protection, tech support, streaming) and **demographics** (gender, senior citizen, partner, dependents). Explore them in *The problem* and *Data*.'))
FAQ.append((['hola', 'buenas', 'hello', 'hi', 'gracias', 'thanks'],
            '¡Hola! Puedo contarte sobre los datos, quiénes abandonan más, cómo se compararon los modelos, qué significan las métricas o cómo usar cada sección del dash. ¿Qué te gustaría saber?',
            'Hi! I can tell you about the data, who churns the most, how the models were compared, what the metrics mean or how to use each section of the dash. What would you like to know?'))

SUGGESTIONS = [N('¿Qué clientes abandonan más?'), N('¿Cuál fue el mejor modelo?'), N('¿Qué es SMOTE y por qué se usó?'), N('¿Cómo uso el simulador?'), N('¿Cómo descargo los datos?')]

SYSTEM = """Eres el asistente del dashboard "Señal perdida", sobre el proyecto Telco Customer Churn de Laura S. Rivera Figueredo y Natalý A. Cárdenas Izaquita (Universidad del Norte).
Respondes preguntas de cualquier visitante sobre el proyecto, sus resultados y cómo usar el dashboard.

Hechos del proyecto:
- Datos: Telco Customer Churn de IBM, 7,043 clientes, 21 columnas, 26.5 % de abandono. Partición estratificada 80/20 (5,634 entrenamiento, 1,409 test).
- Tasas de abandono más altas: cheque electrónico 45.7 %, contrato mes a mes 42.7 %, fibra óptica 42.1 %, sin seguridad en línea 41.9 %, sin soporte técnico 41.8 %. Con contrato a dos años: 2.9 %.
- Diseño: 7 familias de modelos × 4 balanceos × 4 optimizadores = 112 configuraciones, validación cruzada anidada (5 folds externos, 3 internos), F1 de la clase de abandono como métrica principal.
- Resultados: Random Forest + SMOTE F1 0.638 ± 0.023 (AUC 0.844), XGBoost + SMOTE 0.638, regresión logística 0.634; sin diferencias estadísticamente detectables. SMOTE mejoró 6 de 7 familias. La calibración sigmoide mejoró Brier y ECE en 80 de 96 configuraciones.
- SHAP: contrato, servicio de internet y antigüedad son las variables más importantes. Las atribuciones no son causales.
- Dashboard: secciones El problema, Datos (con filtros y descarga de CSV en Explorador), Preparación, Arena de modelos, Simulador (perfil de riesgo, clientes reales, umbral de alerta), ¿Por qué se van?, Conclusiones y Autoras. Botones ES/EN y modo claro/oscuro arriba a la derecha.
- Jupyter Book: """ + BOOK + """

Reglas:
- Responde en el idioma que te indiquen, en 2 a 5 frases, con markdown sencillo. Cita cifras solo si aparecen en estos hechos o en el contexto recibido.
- Si la pregunta no tiene que ver con el proyecto o el dashboard, dilo con amabilidad y sugiere qué se puede preguntar.
- Si el contexto no alcanza para responder, dilo y recomienda la sección del dashboard o del Jupyter Book donde buscar."""


def _chunks():
    """Fragmentos del texto del libro (en español) para recuperar contexto."""
    out = []
    for chapter, sections in a.CHAPTERS.items():
        for s in sections:
            text = re.sub(r'\s+', ' ', re.sub(r'[#*`|>$\\]', ' ', s['text'])).strip()
            for i in range(0, len(text), 700):
                out.append((f'{chapter} · {s["title"]}', text[i:i + 900]))
    return out


def _vectorizer(): return TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), strip_accents='unicode', lowercase=True, sublinear_tf=True)


@lru_cache(maxsize=1)
def index():
    # Preguntas de ejemplo: cada una se compara por separado y la entrada toma su mejor coincidencia.
    questions = [(i, q) for i, (qs, _, _) in enumerate(FAQ) for q in qs]
    qvec = _vectorizer(); qm = qvec.fit_transform([q for _, q in questions])
    chunks = _chunks()
    cvec = _vectorizer(); cm = cvec.fit_transform([t + ' ' + c for t, c in chunks])
    return qvec, qm, np.array([i for i, _ in questions]), cvec, cm, chunks


def retrieve(question):
    qvec, qm, owner, cvec, cm, chunks = index()
    qs = (qm @ qvec.transform([question]).T).toarray().ravel()
    fs = np.zeros(len(FAQ)); np.maximum.at(fs, owner, qs)
    cs = (cm @ cvec.transform([question]).T).toarray().ravel()
    return [(fs[i], i) for i in np.argsort(-fs)[:3]], [(cs[i], chunks[i]) for i in np.argsort(-cs)[:3]]


def local_answer(question, lang):
    faqs, chunks = retrieve(question)
    score, i = faqs[0]
    if score >= .5:
        return FAQ[i][2 if lang == 'en' else 1]
    best, (title, text) = chunks[0]
    if best >= .12:
        intro = 'This is what the book says (in Spanish), in' if lang == 'en' else 'Esto dice el libro en'
        return f'{intro} *{title}*:\n\n> {text[:380].strip()}…'
    return T('No encontré una respuesta a eso en el proyecto. Puedes preguntarme por los datos, los factores de abandono, los modelos, las métricas o cómo usar cada sección del dash.')


@lru_cache(maxsize=1)
def client():
    if not os.environ.get('ANTHROPIC_API_KEY'): return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


def llm_answer(question, history, lang):
    """Respuesta redactada por Claude con el contexto recuperado; None si no hay API o falla."""
    c = client()
    if c is None: return None
    import anthropic
    faqs, chunks = retrieve(question)
    context = '\n'.join(f'- {FAQ[i][1]}' for s, i in faqs if s > .08) + '\n' + '\n'.join(f'- [{t}] {x}' for s, (t, x) in chunks if s > .08)
    language = 'inglés' if lang == 'en' else 'español'
    messages = [{'role': m['role'], 'content': m['text']} for m in history[-6:]]
    messages.append({'role': 'user', 'content': f'<contexto>\n{context}\n</contexto>\n\nResponde en {language}.\n\nPregunta: {question}'})
    try:
        r = c.beta.messages.create(model=MODEL, max_tokens=1024, system=SYSTEM, messages=messages, cache_control={'type': 'ephemeral'},
                                   output_config={'effort': 'low'}, betas=['server-side-fallback-2026-07-01'], fallbacks='default')
    except (anthropic.APIStatusError, anthropic.APIConnectionError):
        return None
    if r.stop_reason == 'refusal': return None
    text = ''.join(b.text for b in r.content if b.type == 'text').strip()
    return text or None


def answer(question, history, lang):
    return llm_answer(question, history, lang) or local_answer(question, lang)


# ---------------------------------------------------------------- interfaz
def layout():
    return html.Div([
        dcc.Store(id='chat-log', storage_type='session', data=[]),
        html.Div([
            html.Div([html.Span(icon('signal'), className='chat-avatar'), html.Div([html.Strong(id='chat-title'), html.Small(id='chat-sub')]),
                      html.Button(icon('close'), id='chat-close', className='icon-btn', n_clicks=0)], className='chat-head'),
            html.Div(id='chat-quota', className='chat-quota'),
            dcc.Loading(html.Div(id='chat-messages', className='chat-messages'), type='dot', color='#1BA7C4', parent_className='chat-loading'),
            html.Div([dcc.Input(id='chat-input', type='text', debounce=False, autoComplete='off', maxLength=400, n_submit=0),
                      html.Button(icon('send'), id='chat-send', className='chat-send', n_clicks=0)], className='chat-form')], id='chat-panel', className='chat-panel'),
        html.Button(icon('chat'), id='chat-fab', className='chat-fab', n_clicks=0)], id='chat', className='chat')


def bubble(role, text): return html.Div(dcc.Markdown(text, link_target='_blank'), className='msg ' + role)


def register(app):
    app.clientside_callback("function(a, b){ var o = document.getElementById('chat'); if(!o) return ''; var open = !o.classList.contains('open');"
                            " if (window.dash_clientside.callback_context.triggered[0].prop_id.indexOf('chat-close') === 0) open = false; return open ? 'chat open' : 'chat'; }",
                            Output('chat', 'className'), Input('chat-fab', 'n_clicks'), Input('chat-close', 'n_clicks'), prevent_initial_call=True)

    app.clientside_callback("function(c){ setTimeout(function(){ var m = document.getElementById('chat-messages'); if (m) m.scrollTop = m.scrollHeight; }, 60); return ''; }",
                            Output('chat-messages', 'title'), Input('chat-messages', 'children'))

    @app.callback(Output('chat-log', 'data'), Output('chat-input', 'value'),
                  Input('chat-send', 'n_clicks'), Input('chat-input', 'n_submit'), Input({'type': 'chat-sug', 'index': ALL}, 'n_clicks'),
                  State('chat-input', 'value'), State('chat-log', 'data'), State('lang', 'data'), prevent_initial_call=True)
    def ask(_send, _submit, sug_clicks, text, log, lang):
        set_lang(lang)
        trig = ctx.triggered_id
        if isinstance(trig, dict):
            if not any(sug_clicks): return no_update, no_update
            text = T(SUGGESTIONS[trig['index']])
        text = (text or '').strip()
        log = log or []
        if not text or sum(m['role'] == 'user' for m in log) >= MAX_QUESTIONS: return no_update, ''
        reply = answer(text, log, lang)
        return log + [{'role': 'user', 'text': text}, {'role': 'assistant', 'text': reply}], ''

    @app.callback(Output('chat-messages', 'children'), Output('chat-quota', 'children'), Output('chat-title', 'children'), Output('chat-sub', 'children'),
                  Output('chat-input', 'placeholder'), Input('chat-log', 'data'), Input('lang', 'data'))
    def render(log, lang):
        set_lang(lang)
        log = log or []
        left = MAX_QUESTIONS - sum(m['role'] == 'user' for m in log)
        hello = T('¡Hola! Soy el asistente de Señal perdida. Pregúntame lo que quieras sobre el proyecto o sobre cómo usar el dash. Tienes {n} preguntas disponibles.').format(n=MAX_QUESTIONS)
        msgs = [bubble('assistant', hello)]
        if not log:
            msgs.append(html.Div([html.Button(T(s), id={'type': 'chat-sug', 'index': i}, className='chat-chip', n_clicks=0) for i, s in enumerate(SUGGESTIONS)], className='chat-chips'))
        msgs += [bubble(m['role'], m['text']) for m in log]
        if left <= 0: msgs.append(bubble('assistant', T('Llegaste al límite de preguntas de esta sesión. Abre el dash en una pestaña nueva para empezar otra conversación.')))
        quota = [html.Span(T('{n} preguntas disponibles').format(n=left)), html.Span([html.I(className='qdot' + (' used' if i >= left else '')) for i in range(MAX_QUESTIONS)], className='qdots')]
        mode = T('Respuestas con Claude') if client() else T('Respuestas del proyecto')
        return msgs, quota, T('Asistente Señal'), T('Datos: proyecto Telco Churn') + ' · ' + mode, T('Pregunta sobre el abandono de clientes…')
