"""Señal perdida: dashboard interactivo del proyecto Telco Customer Churn de Laura Rivera y Natalý Cárdenas.

Cada sección corresponde a un capítulo del Jupyter Book y usa los mismos tipos de gráfica que el libro.
"""
import json
import os
import re
from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
from flask import send_from_directory
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import analytics as a
import chatbot
import eda
from i18n import EN, T, N, set_lang, get_lang
from icons import icon

BOOK = 'https://nataly-cardenas.github.io/Telco_Churn_Entregable/'
REPO = 'https://github.com/Nataly-Cardenas/Telco_Churn_Entregable'
DASH_REPO = 'https://github.com/Nataly-Cardenas/Dash_Telco_Churn'

# ---------------------------------------------------------------- paleta
# Un solo color para todas las series; el segundo color solo resalta el mejor resultado, la clase de abandono
# o el dato que la gráfica quiere señalar. Son los mismos tonos del Jupyter Book.
BASE, HOT, GREY = '#2E86AB', '#E4572E', '#9AA5B1'
LOW, MID, HIGH = BASE, GREY, HOT                       # niveles de riesgo del simulador
SEQ = [[0, 'rgba(46,134,171,.06)'], [1, BASE]]          # escala de una sola tinta
DIVERGING = [[0, BASE], [.5, 'rgba(154,165,177,.15)'], [1, HOT]]
GRID = 'rgba(139,147,167,.18)'


def highlight(values, best=None):
    """Colores de barras: todas en BASE y la mejor (o la indicada) en HOT."""
    values = list(values)
    if best is None: best = int(np.nanargmax(values)) if values else -1
    best = best if isinstance(best, (list, set, tuple)) else [best]
    return [HOT if i in best else BASE for i in range(len(values))]


NAV = [('introduccion', N('Introducción'), 'signal'), ('eda', N('Análisis exploratorio'), 'chart'), ('preprocesamiento', N('Preprocesamiento'), 'flask'),
       ('modelado', N('Modelado avanzado'), 'trophy'), ('simulador', N('Simulador'), 'sliders'), ('conclusiones', N('Conclusiones'), 'flag'), ('autoras', N('Autoras'), 'users')]
TITLES = {'inicio': N('Inicio'), **{k: t for k, t, _ in NAV}}
CHAPTER = {'introduccion': 'intro', 'eda': 'EDA_', 'preprocesamiento': 'PREPROCESAMIENTO', 'modelado': 'MODELADO_AVANZADO'}
DATA_PAGES = {'eda'}

app = Dash(__name__, suppress_callback_exceptions=True, title='Señal perdida · Telco Customer Churn',
           external_stylesheets=['https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap'],
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
                      {'name': 'description', 'content': 'Dashboard interactivo para entender y anticipar el abandono de clientes de telecomunicaciones.'}])
server = app.server
# El tema se aplica antes de pintar para evitar un destello del tema equivocado.
app.index_string = app.index_string.replace('<html>', '<html lang="es">').replace('{%favicon%}', '{%favicon%}<script>try{var t=JSON.parse(localStorage.getItem("theme"));if(t)document.documentElement.dataset.theme=t;var l=JSON.parse(localStorage.getItem("lang"));if(l)document.documentElement.lang=l;}catch(e){}</script>')


@server.route('/evidence/<path:filename>')
def evidence(filename):
    return send_from_directory(a.DATA / 'interpretabilidad', filename)


@server.route('/healthz')
def health(): return {'status': 'ok', 'clients': len(a.RAW), 'experiments': len(a.SUMMARY)}


# ---------------------------------------------------------------- datos del modelado
@lru_cache(maxsize=1)
def oof(): return pd.read_csv(a.DATA / 'oof_mejores.csv')
@lru_cache(maxsize=1)
def anytime(): return pd.read_csv(a.DATA / 'anytime.csv')
@lru_cache(maxsize=1)
def diversity(): return pd.read_csv(a.DATA / 'diversidad_generaciones.csv')
@lru_cache(maxsize=1)
def shap_sample(): return pd.read_csv(a.DATA / 'shap_muestra.csv')
@lru_cache(maxsize=1)
def baseline(): return pd.read_csv(a.DATA / 'baseline_test.csv')


BEST_BY_FAMILY = {'knn': 'knn__smote__bayesian_tpe', 'naive_bayes': 'naive_bayes__none__random', 'logistic': 'logistic__smote__grid', 'decision_tree': 'decision_tree__smote__bayesian_tpe',
                  'random_forest': 'random_forest__smote__bayesian_tpe', 'xgboost': 'xgboost__smote__random', 'svm': 'svm__smote__genetic'}


# ---------------------------------------------------------------- traducción de datos
def cat(v):
    """Nombre visible de una categoría del dataset (vienen en inglés)."""
    if get_lang() == 'en': return {0: 'No', 1: 'Yes'}.get(v, v) if not isinstance(v, str) else v
    return a.ES.get(v, v) if isinstance(v, str) else {0: 'No', 1: 'Sí'}.get(v, v)
def label(var): return T(a.LABELS.get(var, var))
def model_name(m): return T(a.MODELS.get(m, m))
def balance_name(b): return T(a.BALANCES.get(b, b))
def optimizer_name(o): return T(a.OPTIMIZERS.get(o, o))
def config_name(e): m, b, o = e.split('__'); return f'{model_name(m)} · {balance_name(b)} · {optimizer_name(o)}'
def outcome(d): return d.target.map({0: T('Permanece'), 1: T('Abandona')})
def outcome_colors(): return {T('Permanece'): BASE, T('Abandona'): HOT}
def churn_label(v): return {'No': T('Permanece'), 'Yes': T('Abandona')}[v]


# ---------------------------------------------------------------- componentes
def md(text): return dcc.Markdown(text, mathjax=True, className='prose')
def card(*children, cls=''): return html.Section(list(children), className='card ' + cls)
def note(title, text, cls=''): return html.Div([html.Strong(title), html.P(text)], className='note ' + cls)
def grid(*kids, cols=2): return html.Div(list(kids), className=f'grid g{cols}')
def options(values, labels=None): return [{'label': (labels or {}).get(v, cat(v)), 'value': v} for v in values]
def dropdown(id, values, value, labels=None, multi=False): return dcc.Dropdown(options(values, labels), value, id=id, clearable=False, multi=multi, searchable=multi or len(values) > 8)
def field(text, control, id=None): return html.Div([html.Label(text, id=id) if id else html.Label(text), control], className='field')
def eyebrow(text): return html.Span(text, className='eyebrow')
def counter(value, decimals=0, suffix='', prefix=''):
    return html.Span(className='count', **{'data-count': str(value), 'data-decimals': str(decimals), 'data-suffix': suffix, 'data-prefix': prefix})
def kpi(label_, value, sub='', accent='', decimals=0, suffix='', prefix=''):
    v = counter(value, decimals, suffix, prefix) if isinstance(value, (int, float)) else value
    return html.Div([html.Span(label_, className='k-label'), html.Strong(v, className='k-value ' + accent), html.Small(sub)], className='kpi')
def kpis(*items, cls=''): return html.Div(list(items), className='kpis ' + cls)
def details(summary, *children): return html.Details([html.Summary(summary), *children], className='details')
def subhead(title, sub=None): return html.Div([html.H2(title), html.P(sub, className='muted') if sub else None], className='section-head')


def tt(d):
    """Traduce encabezados y textos de una tabla."""
    d = d.copy(); d.columns = [T(str(c)) for c in d.columns]
    for c in d.columns:
        if d[c].dtype == object: d[c] = d[c].map(lambda v: T(v) if isinstance(v, str) else v)
    return d


def table(d, id=None, **kw):
    args = {'id': id} if id else {}
    d = d.copy()
    for c in d.columns:
        if d[c].dtype == bool: d[c] = d[c].map({True: T('Sí'), False: 'No'})
    return dash_table.DataTable(data=d.round(4).replace({np.nan: None}).to_dict('records'), columns=[{'name': str(c), 'id': str(c)} for c in d.columns],
                                page_size=kw.pop('page_size', 10), sort_action='native', filter_action=kw.pop('filter_action', 'none'), style_table={'overflowX': 'auto'},
                                style_header={'backgroundColor': 'var(--table-head)', 'color': 'var(--text)', 'border': 'none', 'fontWeight': '600'},
                                style_cell={'backgroundColor': 'var(--table-cell)', 'color': 'var(--soft)', 'border': 'none', 'borderBottom': '1px solid var(--line)', 'padding': '10px',
                                            'textAlign': 'left', 'fontFamily': 'Inter, sans-serif', 'fontSize': 13.5, 'maxWidth': 340, 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                                style_filter={'backgroundColor': 'var(--table-head)', 'color': 'var(--text)'}, **args, **kw)


def style(f, height=360, legend=True, margin=None):
    # Los colores de texto y rejilla se ajustan al tema desde CSS (assets/style.css, sección "gráficas").
    f.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter, sans-serif', 'color': '#8792a8', 'size': 13.5}, colorway=[BASE, GREY, HOT],
                    margin=margin or {'l': 20, 'r': 20, 't': 36, 'b': 30}, height=height, showlegend=legend,
                    legend={'orientation': 'h', 'y': 1.14, 'title_text': '', 'bgcolor': 'rgba(0,0,0,0)'},
                    transition={'duration': 600, 'easing': 'cubic-in-out'},
                    hoverlabel={'bgcolor': '#0b1426', 'bordercolor': BASE, 'font': {'family': 'Inter', 'color': '#eaf2ff', 'size': 13}})
    f.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID); f.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return f


def graph(f, height=360, legend=True, margin=None):
    return dcc.Graph(figure=style(f, height, legend, margin), style={'height': height, 'width': '100%'},
                     config={'displaylogo': False, 'responsive': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


def chart(title, f, insight='', height=360, legend=True):
    return card(html.H3(title, className='chart-title'), graph(f, height, legend), html.P(insight, className='insight') if insight else None)


def page_head(key, lead):
    chapter = CHAPTER.get(key)
    link = html.A([icon('book', 'sm'), T('Ver este capítulo en el Jupyter Book'), icon('arrow', 'sm')], href=BOOK + chapter + '.html', target='_blank', className='chip-link') if chapter else None
    return html.Div([eyebrow('TELCO CUSTOMER CHURN'), html.H1(T(TITLES[key])), html.P(lead, className='lead'), link], className='page-head')


def tabs(id, items):
    return dcc.Tabs([dcc.Tab(children=html.Div(content, className='tab-body'), label=lbl, value=str(i), className='tab', selected_className='tab on')
                     for i, (lbl, content) in enumerate(items)], value='0', id=id, className='tabs', parent_className='tabs-wrap')


def empty(): return note(T('Sin clientes en este segmento'), T('Amplía los filtros para volver a calcular las visualizaciones.'))
def scope(d): return html.Div([html.Span(className='live-dot'), ' ' + T('{n} clientes en la selección').format(n=f'{len(d):,}')], className='scope')


def flags(c):
    """Señales del perfil asociadas a más (▲) o menos (▼) abandono en el EDA."""
    out = []
    if c.Contract == 'Month-to-month': out.append('▲ ' + T('Contrato mes a mes'))
    if c.Contract == 'Two year': out.append('▼ ' + T('Contrato a dos años'))
    if c.InternetService == 'Fiber optic': out.append('▲ ' + T('Fibra óptica'))
    if c.PaymentMethod == 'Electronic check': out.append('▲ ' + T('Paga con cheque electrónico'))
    if c.tenure <= 12: out.append('▲ ' + T('Menos de un año como cliente'))
    if c.tenure > 48: out.append('▼ ' + T('Más de 4 años como cliente'))
    if c.TechSupport == 'No': out.append('▲ ' + T('Sin soporte técnico'))
    if c.OnlineSecurity == 'Yes': out.append('▼ ' + T('Con seguridad en línea'))
    return out[:4]


@lru_cache(maxsize=1)
def cover_rows():
    """30 clientes reales del test, mezclando riesgo alto, medio y bajo según el Random Forest finalista."""
    p = a.predictions(a.FINALISTS[0]).set_index('row_id')
    rng = np.random.default_rng(7)
    picks = list(rng.choice(p.index[p.probability >= .6], 12, replace=False)) + list(rng.choice(p.index[p.probability.between(.3, .6)], 8, replace=False)) \
        + list(rng.choice(p.index[p.probability < .3], 10, replace=False))
    rng.shuffle(picks)
    return [(int(r), float(p.loc[r, 'probability'])) for r in picks]


def cover_customers():
    out = []
    for r, prob in cover_rows():
        c = a.RAW.loc[r]
        out.append({'id': c.customerID, 'contract': cat(c.Contract), 'internet': cat(c.InternetService), 'payment': cat(c.PaymentMethod),
                    'tenure': int(c.tenure), 'monthly': float(c.MonthlyCharges), 'prob': round(prob, 3), 'churn': int(c.target), 'flags': flags(c)})
    return out


def phone():
    text = {'months': T('meses'), 'risk': {'alto': T('Riesgo alto'), 'medio': T('Riesgo medio'), 'bajo': T('Riesgo bajo')},
            'churned': T('En la realidad: abandonó'), 'stayed': T('En la realidad: permaneció'),
            'toast': {'alto': T('Alerta de retención enviada'), 'medio': T('Cliente en observación'), 'bajo': T('Cliente estable')}}
    return html.Div(html.Div([
        html.Div([html.Span('9:41'), html.Span([html.B(), html.B(), html.B(), html.B()], className='sig'), html.Span('5G', className='bat')], className='phone-status'),
        html.Div([
            html.Div([html.Span(T('TELCO · RADAR DE CLIENTES'), className='pc-app'), html.Span(T('EN VIVO'), className='pc-live')], className='between'),
            html.Div([html.Small(T('Cliente')), html.Strong(className='pc-id')], className='pc-row'),
            html.Div([html.Span(className='pc-contract pill'), html.Span(className='pc-internet pill'), html.Span(className='pc-payment pill')], className='pc-pills'),
            html.Div([html.Div([html.Small(T('Antigüedad')), html.Strong(className='pc-tenure')]), html.Div([html.Small(T('Cargo mensual')), html.Strong(className='pc-monthly')])], className='pc-grid'),
            html.Div([html.Small(T('Señales del perfil')), html.Ul(className='pc-flags')], className='pc-signals'),
            html.Div([html.Div(html.Div([html.Strong(className='ring-pct'), html.Small(T('prob. de abandono'))], className='ring-label'), className='ring'),
                      html.Div([html.Strong(className='pc-level'), html.Small(T('Random Forest + SMOTE · test histórico')), html.Small(className='pc-truth')], className='pc-verdict')], className='pc-risk'),
            html.Div([html.Span(className='toast-dot'), html.Span(className='toast-text')], className='toast')], className='phone-body')],
        id='phone', className='phone', **{'data-customers': json.dumps(cover_customers()), 'data-text': json.dumps(text)}), className='phone-stage')


def cover():
    return html.Div([
        html.Canvas(id='net-canvas', className='net-canvas'),
        html.Div(className='cover-glow'),
        html.Div([
            html.Div([
                eyebrow('TELCO CUSTOMER CHURN · UNIVERSIDAD DEL NORTE'),
                html.H1([T('Señal'), html.Br(), html.Span(T('perdida'), className='glitch', **{'data-text': T('perdida')})], className='cover-title'),
                html.P([T('Uno de cada cuatro clientes '), html.B(T('cuelga para siempre')), T('. Descubre quiénes son, por qué se van y cómo anticiparlo con datos.')], className='cover-lead'),
                html.Div([
                    html.Div([counter(7043), html.Small(T('clientes analizados'))]),
                    html.Div([counter(26.5, 1, ' %'), html.Small(T('abandonan el servicio'))], className='hot'),
                    html.Div([counter(112), html.Small(T('modelos en competencia'))]),
                    html.Div([counter(0.638, 3), html.Small(T('F1 del mejor modelo'))])], className='cover-stats'),
                html.Div([dcc.Link([T('Entrar al dashboard'), ' →'], href='#introduccion', className='btn primary'),
                          html.A([T('Jupyter Book'), icon('arrow', 'sm')], href=BOOK, target='_blank', className='btn ghost')], className='cover-cta'),
                html.Div([icon('cursor'), html.Span([T('Los puntos rojos son clientes a punto de irse: '), html.B(T('pasa el cursor (o el dedo) sobre ellos para retenerlos.'))]),
                          html.Span([T('Perdidos: '), html.B('0', id='cover-lost'), ' · ', T('Retenidos por ti: '), html.B('0', id='cover-saved')], className='score')], className='cover-hint'),
            ], className='cover-text'),
            phone()], className='cover-inner'),
        html.Div('Laura S. Rivera Figueredo · Natalý A. Cárdenas Izaquita', className='cover-foot')], className='cover')


# ================================================================ 1. INTRODUCCIÓN
FLIP = [('target', N('Abandono y clasificación binaria'), N('El churn es la cancelación del servicio. Se estima para cada cliente la probabilidad P(abandono | x) y se separa a quienes se van (1) de quienes permanecen (0).')),
        ('chart', N('Análisis exploratorio'), N('Antes de modelar se mide la asociación de cada variable con el abandono: correlación punto-biserial, V de Cramér y Spearman, con corrección de Holm.')),
        ('scale', N('Desbalance de clases'), N('Solo el 26.5 % abandona. SMOTE y ADASYN crean ejemplos sintéticos de la minoría para que el modelo no la ignore.')),
        ('block', N('Fuga de información'), N('Imputar, codificar, escalar y balancear solo con datos de entrenamiento; si no, las métricas resultan optimistas.')),
        ('tree', N('Modelos de clasificación'), N('Regresión logística como modelo base, frente a KNN, Naive Bayes, árbol de decisión, Random Forest, XGBoost y SVM.')),
        ('repeat', N('Validación cruzada anidada'), N('Un bucle interno elige hiperparámetros con Grid, Random, TPE o un algoritmo genético; uno externo mide el desempeño con datos que no intervinieron.')),
        ('sigma', N('Métricas'), N('F1 = 2TP / (2TP + FP + FN) equilibra detectar abandonos y evitar falsas alertas; AUC-ROC mide qué tan bien se ordena el riesgo. Friedman, Nemenyi y DeLong comparan modelos.')),
        ('bulb', N('Interpretabilidad'), N('SHAP reparte cada predicción entre las variables y LIME explica casos individuales. Describen al modelo, no causas del abandono.'))]
OBJECTIVES = [(N('Describir el problema y los datos'), N('Caracterizar el conjunto, su calidad y el desbalance de la variable objetivo.')),
              (N('Reservar un conjunto de prueba'), N('Separar el 20 % de los clientes con una partición estratificada antes del análisis.')),
              (N('Identificar los perfiles de abandono'), N('Analizar cada variable y su asociación con el abandono para encontrar los segmentos con más cancelaciones.')),
              (N('Preparar los datos sin fuga de información'), N('Definir faltantes, codificación, escalado y variables derivadas ajustándolos solo con entrenamiento.')),
              (N('Establecer un modelo de referencia'), N('Entrenar una regresión logística y cuantificar sus aciertos y errores.')),
              (N('Comparar alternativas de forma controlada'), N('Evaluar 112 configuraciones con validación cruzada anidada y el F1 de la clase de abandono.')),
              (N('Explicar las predicciones'), N('Identificar con SHAP y LIME las variables que más influyen, sin atribuirles un efecto causal.')),
              (N('Poner los resultados a disposición'), N('Presentar el análisis en un Jupyter Book, un paper y este tablero interactivo.'))]
GROUPS = [(N('Identificador'), ['customerID']), (N('Demografía'), ['gender', 'SeniorCitizen', 'Partner', 'Dependents']),
          (N('Servicios'), ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']),
          (N('Cuenta'), ['tenure', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']), (N('Objetivo'), ['Churn'])]


def people_row():
    leaving = {2, 7, 11, 16, 19}   # 5 de 20 ≈ 26.5 % de abandono
    return html.Div([html.Span(className='person' + (' leaving' if i in leaving else ''), style={'animationDelay': f'{(i % 5) * .9}s'}) for i in range(20)], className='people')


def introduccion(d):
    variables = pd.DataFrame([{'Grupo': T(g), 'Variable': v, 'Significado': label(v) if v in a.LABELS else T('Código del cliente') if v == 'customerID' else T('Abandono durante el último mes'),
                               'Tipo': T('Numérica') if v in a.NUMERIC else T('Identificador') if v == 'customerID' else T('Categórica'),
                               'Papel': T('Objetivo') if v == 'Churn' else T('No predictor') if v == 'customerID' else T('Candidato')} for g, vs in GROUPS for v in vs])
    return html.Div([
        page_head('introduccion', T('Perder un cliente cuesta más que conservarlo. Este proyecto busca anticipar quién está por cancelar su servicio para actuar a tiempo.')),
        html.Div([html.Div([eyebrow(T('CONTEXTO DEL PROBLEMA')), html.H2([T('1 de cada 4 clientes '), html.Span(T('se desconecta'), className='txt-hot')]),
                            html.P(T('El conjunto Telco Customer Churn de IBM reúne servicios contratados, datos de cuenta y perfil demográfico de cada cliente, '
                                     'junto con si canceló en el último mes. Dos dificultades: los que se van son minoría, y los datos son una foto de un momento, '
                                     'así que los hallazgos son asociaciones y no causas.')), people_row()], className='hero-text'),
                  kpis(kpi(T('CLIENTES'), 7043, T('una fila por cliente')), kpi(T('ABANDONOS'), 1869, T('cancelaron en el último mes')),
                       kpi(T('TASA DE ABANDONO'), 26.5, T('la clase minoritaria'), 'hot', 1, ' %'), kpi(T('VARIABLES'), 19, T('predictoras candidatas')), cls='k2')], className='hero'),
        subhead(T('Objetivo general')),
        card(html.P(T('Identificar a los clientes de una empresa de telecomunicaciones con mayor riesgo de abandonar el servicio y los factores asociados a ese abandono, '
                      'construyendo y comparando de forma reproducible modelos de clasificación que detecten la mayor cantidad posible de clientes en riesgo sin generar un exceso de falsas alertas.'), className='big-text'), cls='glow'),
        subhead(T('Objetivos específicos')),
        html.Div([html.Div([html.Span(f'{i + 1:02d}', className='obj-n'), html.H4(T(t)), html.P(T(s))], className='obj') for i, (t, s) in enumerate(OBJECTIVES)], className='objectives'),
        subhead(T('Marco teórico'), T('Pasa el cursor (o toca) cada tarjeta para voltearla.')),
        html.Div([html.Div(html.Div([html.Div([html.Span(icon(i), className='flip-icon'), html.H4(T(t))], className='flip-front'), html.Div([html.H4(T(t)), html.P(T(s))], className='flip-back')],
                                    className='flip-inner'), className='flip', tabIndex=0) for i, t, s in FLIP], className='flips'),
        subhead(T('Descripción de las variables del proyecto'), T('Diccionario resumido de los metadatos de Kaggle.')),
        card(table(variables, page_size=21))])


# ================================================================ 2. ANÁLISIS EXPLORATORIO
def eda_page(d):
    if d.empty: return html.Div([page_head('eda', T('Explora el conjunto de datos con los filtros superiores.')), empty()])
    num_opts = {v: v for v in a.NUMERIC}; cat_opts = {v: v for v in a.CATEGORICAL}
    etl = [kpis(kpi(T('FILAS ORIGINALES'), 7043, T('21 columnas')), kpi(T('ENTRENAMIENTO'), 5634, T('80 % · aquí se hace el EDA')), kpi(T('TEST'), 1409, T('20 % · reservado')),
                kpi(T('DUPLICADOS'), int(d[a.RAW.columns[:21]].duplicated().sum()), T('filas repetidas en la selección'))),
           grid(chart(T('División train / test estratificada'), split_figure(), T('El test se separa primero y conserva la proporción de abandono (26.5 %) en ambas particiones.'), 320),
                card(html.H3(T('Estructura original'), className='chart-title'), table(tt(structure_table()), page_size=8),
                     html.P(T('TotalCharges llega como texto: los espacios en blanco se convierten en valores faltantes y la columna pasa a numérica.'), className='insight')))]
    univariado = [grid(chart(T('3.1 Variable objetivo'), target_figure(d), T('Clientes que permanecen y que abandonan: el desbalance obliga a usar métricas de la clase positiva.'), 340, False),
                       card(html.H3(T('3.2.1 Estadísticas descriptivas'), className='chart-title'), table(tt(eda.numeric_summary(d))))),
                  subhead(T('3.2 Variables numéricas')),
                  card(field(T('Variable'), dropdown('uni-num', a.NUMERIC, 'tenure', num_opts)), html.Div(id='uni-num-output'), cls='controls-card'),
                  subhead(T('3.3 Variables categóricas')),
                  card(field(T('Variable'), dropdown('uni-cat', ['__all__'] + a.CATEGORICAL, '__all__', {'__all__': T('Todas las variables'), **cat_opts})), html.Div(id='uni-cat-output'), cls='controls-card'),
                  subhead(T('3.4 Outliers y sesgos')),
                  card(table(tt(eda.outliers(d))), html.P(T('Con el criterio IQR ninguna variable numérica presenta valores atípicos; TotalCharges tiene asimetría positiva. No se eliminan observaciones.'), className='insight'))]
    bivariado = [subhead(T('4.1 Variable objetivo vs variables numéricas')),
                 card(field(T('Variable'), dropdown('bi-num', a.NUMERIC, 'tenure', num_opts)), html.Div(id='bi-num-output'), cls='controls-card'),
                 html.Div(id='bi-pb-output'),
                 subhead(T('4.2 Variable objetivo vs variables categóricas')),
                 card(field(T('Variable'), dropdown('bi-cat', a.CATEGORICAL, 'Contract', cat_opts)), html.Div(id='bi-cat-output'), cls='controls-card'),
                 html.Div(id='bi-holm-output')]
    multivariado = [html.Div(id='multi-output')]
    faltantes = [html.Div(id='missing-output')]
    resumen = [eda_summary()]
    explorador = [card(field(T('Columnas visibles'), dropdown('explorer-cols', list(a.RAW.columns[:21]), ['customerID', 'Contract', 'InternetService', 'tenure', 'MonthlyCharges', 'Churn'], {}, multi=True)),
                       html.Div(id='explorer-table'),
                       html.Div([html.Button([icon('download'), T('Descargar CSV')], id='export', className='btn primary', n_clicks=0),
                                 html.P(T('Descarga un solo archivo con la selección actual de los filtros superiores y las columnas elegidas.'), className='muted')], className='export-row'), cls='controls-card')]
    return html.Div([page_head('eda', T('Del archivo original a los patrones de abandono. Todo el análisis se hace sobre el conjunto de entrenamiento; los filtros superiores recalculan las gráficas y tablas.')),
                     scope(d),
                     kpis(kpi(T('CLIENTES'), len(d), T('en la selección')), kpi(T('ABANDONO'), round(d.target.mean() * 100, 2), T('{n} clientes').format(n=f'{int(d.target.sum()):,}'), 'hot', 2, ' %'),
                          kpi(T('ANTIGÜEDAD MEDIA'), round(d.tenure.mean(), 1), T('meses'), '', 1), kpi(T('CARGO MENSUAL MEDIO'), round(d.MonthlyCharges.mean(), 2), T('dólares'), '', 2, '', '$')),
                     tabs('eda-tabs', [(T('1–2 · ETL y partición'), etl), (T('3 · Univariado'), univariado), (T('4 · Bivariado'), bivariado), (T('5 · Multivariado'), multivariado),
                                       (T('6 · Valores faltantes'), faltantes), (T('7–8 · Resumen'), resumen), (T('Explorador de datos'), explorador)])])


def split_figure():
    tr, te = a.RAW.loc[a.TRAIN_IDS], a.RAW.loc[a.TEST_IDS]
    g = pd.DataFrame({T('Partición'): [T('Entrenamiento')] * 2 + ['Test'] * 2, 'Churn': [T('Permanece'), T('Abandona')] * 2,
                      'n': [int((tr.target == 0).sum()), int(tr.target.sum()), int((te.target == 0).sum()), int(te.target.sum())]})
    f = px.bar(g, x=T('Partición'), y='n', color='Churn', text='n', color_discrete_map=outcome_colors(), labels={'n': T('Clientes')})
    f.update_traces(textposition='inside'); f.update_xaxes(title='')
    return f


def structure_table():
    r = a.RAW[a.RAW.columns[:21]]
    raw = pd.read_csv(a.DATA / 'WA_Fn-UseC_-Telco-Customer-Churn.csv', nrows=5)
    return pd.DataFrame({'Variable': r.columns, 'Tipo original': [str(raw[c].dtype) for c in r.columns], 'Valores únicos': [r[c].nunique() for c in r.columns]})


def target_figure(d):
    t = eda.target_summary(d); t['Churn'] = t.Churn.map(churn_label)
    f = px.bar(t, x='Churn', y='Clientes', color='Churn', color_discrete_map=outcome_colors(), text=t.apply(lambda r: f"{r['Clientes']:,} · {r['Porcentaje (%)']:.2f}%", axis=1),
               labels={'Clientes': T('Clientes')})
    f.update_traces(textposition='outside'); f.update_yaxes(range=[0, t.Clientes.max() * 1.18]); f.update_xaxes(title='')
    return f


def eda_summary():
    t = a.RAW.loc[a.TRAIN_IDS]
    rates = eda.churn_rates(t); top = rates[rates['Mayor de su variable']].sort_values('Tasa (%)', ascending=False).head(8)
    top['Etiqueta'] = top.Variable + ' = ' + top['Categoría'].astype(str)
    f = go.Figure(go.Bar(x=top['Tasa (%)'][::-1], y=top.Etiqueta[::-1], orientation='h', marker_color=highlight(top['Tasa (%)'][::-1]),
                         text=top['Tasa (%)'][::-1].map('{:.1f}%'.format), textposition='outside'))
    f.add_vline(x=26.54, line_dash='dash', line_color=GREY, annotation_text=T('promedio 26.5 %'), annotation_position='top')
    f.update_xaxes(title=T('Tasa de abandono (%)'), range=[0, 55])
    findings = [N('El abandono es la clase minoritaria (26.5 %): se evaluará con métricas de la clase positiva y se compararán estrategias de balanceo.'),
                N('Los clientes que abandonan tienen menor antigüedad y menor cargo acumulado, pero mayor cargo mensual; tenure es la asociación numérica más fuerte (r = −0.35).'),
                N('Contrato es la variable categórica más asociada (V de Cramér = 0.41), seguida de seguridad en línea y soporte técnico; gender y PhoneService no muestran asociación tras Holm.'),
                N('Ninguna variable numérica es normal, por lo que se usa Spearman; tenure y TotalCharges están muy correlacionadas (ρ = 0.89) y se crea el cargo promedio mensual.'),
                N('Las categorías "sin internet" y "sin teléfono" repiten información y se unifican en "No".'),
                N('Los 8 faltantes de TotalCharges son clientes con antigüedad 0: se reemplazan por 0. No hay valores atípicos según el IQR.')]
    return html.Div([chart(T('Categorías con mayores tasas de abandono'), f, T('La categoría con mayor tasa de cada variable; la línea punteada es el promedio general.'), 420, False),
                     subhead(T('Decisiones que deja el EDA')),
                     html.Div([html.Div([html.Span(f'{i + 1:02d}', className='obj-n'), html.P(T(x))], className='obj') for i, x in enumerate(findings)], className='objectives g3')])


# ================================================================ 3. PREPROCESAMIENTO
STEPS = [('target', N('4.1 Objetivo'), N('Churn Yes/No → 1/0. customerID solo para control.')), ('shield', N('4.2 TotalCharges'), N('Texto a número; los 8 vacíos de train (3 de test) tienen tenure 0 → 0.')),
         ('layers', N('4.3 Ingeniería'), N('“Sin internet / sin teléfono” → “No” y AvgChargePerMonth = TotalCharges / tenure.')), ('table', N('4.4–4.5 Codificar y escalar'), N('4 numéricas con StandardScaler y 16 categóricas con One-Hot: 24 columnas.')),
         ('scale', N('Balancear'), N('Ninguno, pesos, SMOTE o ADASYN; solo dentro del entrenamiento.')), ('network', N('Modelo'), N('Se ajusta en entrenamiento y se evalúa en el fold reservado.'))]
COEFS = [('cat__Contract_Two year', -1.422598), ('num__tenure', -1.153496), ('cat__InternetService_No', -1.079819), ('cat__InternetService_Fiber optic', 1.036427), ('cat__Contract_One year', -0.725538),
         ('num__TotalCharges', 0.486039), ('cat__PaymentMethod_Electronic check', 0.398853), ('cat__OnlineSecurity_Yes', -0.371758), ('cat__StreamingMovies_Yes', 0.348344), ('cat__PaperlessBilling_Yes', 0.334539),
         ('cat__StreamingTV_Yes', 0.323924), ('cat__PhoneService_Yes', -0.31652), ('cat__TechSupport_Yes', -0.304333), ('cat__MultipleLines_Yes', 0.28687), ('num__MonthlyCharges', -0.272618),
         ('cat__Dependents_Yes', -0.231701), ('cat__SeniorCitizen_1', 0.153587), ('cat__OnlineBackup_Yes', -0.110944), ('cat__PaymentMethod_Mailed check', 0.0482), ('num__AvgChargePerMonth', -0.046767),
         ('cat__gender_Male', 0.029063), ('cat__Partner_Yes', 0.019222), ('cat__PaymentMethod_Credit card (automatic)', -0.00774), ('cat__DeviceProtection_Yes', 0.004777)]


def preprocesamiento(d):
    imput = pd.DataFrame({'m': [T('Imputar con 0'), T('Mediana + indicador')], 'auc': [.8459, .8460], 'sd': [.0124, .0125]})
    fi = go.Figure(go.Bar(x=imput.m, y=imput.auc, error_y={'type': 'data', 'array': imput.sd, 'color': GREY}, marker_color=[BASE, GREY], text=imput.auc.map('{:.4f}'.format), textposition='inside'))
    fi.update_yaxes(range=[.8, .87], title='AUC-ROC CV')
    pasos = [card(html.H3(T('El pipeline paso a paso'), className='chart-title'), html.Div([html.Div(className='flow-line'), *[html.Div([html.Span(icon(i), className='st-icon'), html.H4(T(t)), html.P(T(s))], className='station',
                  style={'animationDelay': f'{k * .15}s'}) for k, (i, t, s) in enumerate(STEPS)]], className='pipeline')),
             grid(chart(T('4.2 ¿Cero o mediana para TotalCharges?'), fi, T('Diferencia pareada por fold: +0.0001 ± 0.0001. Imputar con 0 es coherente con clientes sin facturación y no pierde desempeño.'), 320, False),
                  card(html.H3(T('4.3 Ingeniería de características'), className='chart-title'),
                       kpis(kpi(T('CORRELACIÓN CON MONTHLYCHARGES'), .9963, 'AvgChargePerMonth', '', 4), kpi(T('COLUMNAS TRAS CODIFICAR'), 24, T('4 numéricas + 20 indicadores')), cls='k2'),
                       html.P(T('El cargo promedio mensual comparte casi toda la información con el cargo mensual; su aporte se evalúa en la ablación.'), className='insight')))]
    cv = pd.DataFrame({'m': [T('Regresión logística'), T('Etiquetas permutadas'), T('Dummy estratificado'), T('Dummy mayoritaria')], 'auc': [.8454, .4868, .5065, .5], 'sd': [.0123, .0220, .0171, 0]})
    fc = go.Figure(go.Bar(x=cv.auc, y=cv.m, orientation='h', error_x={'type': 'data', 'array': cv.sd, 'color': GREY}, marker_color=highlight(cv.auc), text=cv.auc.map('{:.4f}'.format), textposition='inside'))
    fc.update_xaxes(range=[0, 1], title='AUC-ROC CV (5 folds)'); fc.add_vline(x=.5, line_dash='dot', line_color=GREY)
    validacion = [kpis(kpi('AUC-ROC CV', .8454, '± 0.0123', '', 4), kpi('PR-AUC CV', .6581, '± 0.0191', '', 4), kpi('RECALL CV', .8020, '± 0.0341', '', 4), kpi(T('CONTROL PERMUTADO'), .4868, T('AUC del azar'), '', 4)),
                  chart(T('4.6 Validación cruzada en entrenamiento'), fc, T('El modelo supera con claridad a los controles: con etiquetas permutadas o un clasificador trivial el AUC ronda 0.5.'), 340, False)]
    abl = pd.DataFrame({'c': [T('Sin gender'), T('Sin PhoneService'), T('Sin PhoneService ni MultipleLines'), T('Sin gender ni PhoneService'), T('Sin AvgChargePerMonth')],
                        'd': [.0001, .0002, -.0007, .0004, .0005], 'sd': [.0005, .0003, .0018, .0004, .0003], 'dpr': [-.0003, -.0003, -.0106, -.0000, .0019]})
    fa = go.Figure(go.Bar(x=abl.d, y=abl.c, orientation='h', error_x={'type': 'data', 'array': abl.sd, 'color': GREY}, marker_color=BASE))
    fa.add_vline(x=0, line_color=GREY); fa.update_xaxes(title=T('Δ AUC-ROC frente al modelo completo'), tickformat='.4f')
    ablacion = [chart(T('4.7 Ablación por validación cruzada'), fa, T('Retirar gender, PhoneService o AvgChargePerMonth cambia el AUC menos que la variación entre folds; las variables se conservan.'), 340, False),
                card(table(tt(abl.rename(columns={'c': 'Configuración', 'd': 'Δ AUC', 'sd': 'DE', 'dpr': 'Δ PR-AUC'}))))]
    b = baseline(); cm = confusion_matrix(b.y_true, (b.probability >= .5).astype(int))
    fcm = px.imshow(cm, x=[T('Predice permanencia'), T('Predice abandono')], y=[T('Permanece'), T('Abandona')], text_auto=True, color_continuous_scale=SEQ, aspect='auto')
    fcm.update_coloraxes(showscale=False); fcm.update_traces(textfont_size=18)
    fpr, tpr, _ = roc_curve(b.y_true, b.probability)
    froc = roc_animation([(T('Regresión logística'), fpr, tpr, BASE)])
    co = pd.DataFrame(COEFS, columns=['f', 'c']); co['n'] = co.f.map(pretty); co = co.iloc[::-1]
    fco = go.Figure(go.Bar(x=co.c, y=co.n, orientation='h', marker_color=[HOT if any(k in f for k in ['gender', 'PhoneService']) else BASE for f in co.f]))
    fco.update_xaxes(title=T('Coeficiente (escala estandarizada)')); fco.add_vline(x=0, line_color=GREY)
    test = [kpis(kpi('ACCURACY', .7395, T('frente a 0.7346 de predecir siempre “permanece”'), '', 4), kpi('PRECISION', .5060, '', '', 4), kpi('RECALL', .7834, '', 'hot', 4), kpi('F1', .6149, '', '', 4),
                 kpi('AUC-ROC', .8419, T('IC 95 % [0.8214, 0.8631]'), '', 4), kpi('PR-AUC', .6340, '', '', 4), cls='k6'),
            grid(chart(T('7.2 Matriz de confusión'), fcm, T('Detecta 293 de 374 abandonos, pero 286 de sus 579 alertas son falsas: hay margen de mejora.'), 360, False),
                 chart(T('7.3 Curva ROC'), froc, T('Pulsa play para trazar la curva. AUC = 0.8419 en el test.'), 360)),
            chart(T('7.5 Coeficientes de la regresión logística'), fco, T('Contrato a dos años, antigüedad y no tener internet reducen el riesgo; la fibra óptica lo aumenta. gender y PhoneService (resaltadas) pesan poco.'), 640, False)]
    return html.Div([page_head('preprocesamiento', T('Cómo se transforman los datos antes de modelar, sin espiar el test, y cuánto logra el modelo base.')),
                     tabs('prep-tabs', [(T('4.1–4.5 · Pasos'), pasos), (T('4.6 · Validación cruzada'), validacion), (T('4.7 · Ablación'), ablacion), (T('7 · Evaluación en test'), test)])])


def roc_animation(curves):
    """Curvas ROC que se trazan con un botón de play (mismo tipo de gráfica que el libro)."""
    steps = 24; f = go.Figure()
    for name, fpr, tpr, color in curves:
        f.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines', line={'color': color, 'width': 3 if color == HOT else 2.4}))
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name=T('Azar'), mode='lines', line={'dash': 'dash', 'color': GREY}))
    frames = []
    for k in range(1, steps + 1):
        data = [go.Scatter(x=fpr[:max(2, int(len(fpr) * k / steps))], y=tpr[:max(2, int(len(tpr) * k / steps))]) for _, fpr, tpr, _ in curves]
        frames.append(go.Frame(data=data, traces=list(range(len(curves))), name=str(k)))
    f.frames = frames
    f.update_layout(updatemenus=[{'type': 'buttons', 'showactive': False, 'x': 1, 'y': -.18, 'xanchor': 'right', 'buttons': [
        {'label': '▶', 'method': 'animate', 'args': [None, {'frame': {'duration': 45, 'redraw': False}, 'transition': {'duration': 0}, 'fromcurrent': False}]}]}])
    f.update_xaxes(title=T('Tasa de falsos positivos'), range=[0, 1]); f.update_yaxes(title=T('Tasa de verdaderos positivos (Recall)'), range=[0, 1.02])
    return f


# ================================================================ 4. MODELADO AVANZADO
METRICS = {'f1_positive': N('F1 positivo'), 'recall_positive': 'Recall', 'precision_positive': 'Precision', 'roc_auc': 'ROC-AUC', 'accuracy': 'Accuracy'}


def podium():
    p = [('2', 'XGBoost', 'SMOTE · Random', .638, 'second'), ('1', 'Random Forest', 'SMOTE · TPE', .638, 'first'), ('3', T('Reg. logística'), 'SMOTE · Grid', .634, 'third')]
    return card(eyebrow(T('PODIO · F1 MEDIO EXTERNO')), html.Div([html.Div([html.Span(m, className='medal'), html.Strong(n), html.Small(c), html.Div(counter(v, 3), className='bar-podium')],
                className='pod ' + k) for m, n, c, v, k in p], className='podium'), html.P(T('Empate técnico: la diferencia entre RF y XGBoost es 0.0004, menor que la variación entre folds.'), className='insight'))


def modelado(d):
    fam = {k: model_name(k) for k in a.MODELS}
    resultados = [card(field(T('Familia de modelos (mejor configuración de cada una)'), dropdown('res-family', list(BEST_BY_FAMILY), 'random_forest', fam)), html.Div(id='res-output'), cls='controls-card'),
                  html.Div(id='rank-output-static', children=ranking_families()),
                  card(html.Div([field(T('Familia'), dropdown('model-family', ['Todas'] + list(a.MODELS), 'Todas', {'Todas': T('Todas'), **fam})),
                                 field(T('Balanceo'), dropdown('model-balance', ['Todos'] + list(a.BALANCES), 'Todos', {'Todos': T('Todos'), **{k: balance_name(k) for k in a.BALANCES}})),
                                 field(T('Métrica'), dropdown('model-metric', list(METRICS), 'f1_positive', {k: T(v) for k, v in METRICS.items()}))], className='controls'), cls='controls-card'),
                  html.Div(id='models-output')]
    calibracion = [html.Div(id='cal-output-static', children=calibration_summary())]
    optimizadores = [card(field(T('Familia'), dropdown('opt-family', list(a.MODELS), 'random_forest', fam)), cls='controls-card'), html.Div(id='opt-output')]
    return html.Div([page_head('modelado', T('Siete familias × cuatro balanceos × cuatro optimizadores = 112 configuraciones evaluadas con validación cruzada anidada.')),
                     html.Div([podium(), kpis(kpi(T('CONFIGURACIONES'), 112, '7 × 4 × 4'), kpi(T('UNIDADES EXTERNAS'), 1680, T('112 × 3 semillas × 5 folds')),
                                              kpi('FOLDS', '5 × 3', T('externos × internos')), kpi(T('PRESUPUESTO'), 24, T('evaluaciones por búsqueda')), cls='k2')], className='hero'),
                     tabs('model-tabs', [(T('1 · Resultados por modelo'), resultados), (T('1.8 · Calibración'), calibracion), (T('2 · Optimizadores'), optimizadores),
                                         (T('3 · Comparación estadística'), estadistica()), (T('3.2 · Interpretabilidad'), interpretabilidad())])])


def ranking_families():
    s = a.SUMMARY.set_index('experiment_id').loc[list(BEST_BY_FAMILY.values())].reset_index().sort_values('f1_positive_mean')
    s['n'] = s.experiment_id.map(config_name)
    f = go.Figure(go.Bar(x=s.f1_positive_mean, y=s.n, orientation='h', error_x={'type': 'data', 'array': s.f1_positive_std, 'color': GREY},
                         marker_color=highlight(s.f1_positive_mean), text=s.f1_positive_mean.map('{:.3f}'.format), textposition='inside'))
    f.update_xaxes(range=[.55, .68], title=T('F1 positivo · media de 5 folds externos'))
    return chart(T('Mejor configuración de cada familia'), f, T('Barras: media de los folds externos; bigotes: desviación estándar. Random Forest y XGBoost empatan arriba (0.638).'), 380, False)


def calibration_summary():
    u = a.UNITS[a.UNITS.replicate == 0]
    g = u.groupby('model')[['brier', 'cal_brier', 'ece', 'cal_ece']].mean().reset_index(); g['n'] = g.model.map(model_name)
    fb = go.Figure([go.Bar(x=g.n, y=g.brier, name=T('Nativa'), marker_color=GREY), go.Bar(x=g.n, y=g.cal_brier, name=T('Sigmoide'), marker_color=BASE)])
    fb.update_layout(barmode='group'); fb.update_yaxes(title='Brier')
    fe = go.Figure([go.Bar(x=g.n, y=g.ece, name=T('Nativa'), marker_color=GREY), go.Bar(x=g.n, y=g.cal_ece, name=T('Sigmoide'), marker_color=BASE)])
    fe.update_layout(barmode='group'); fe.update_yaxes(title='ECE')
    improved = int(((u.groupby('experiment_id').cal_brier.mean() < u.groupby('experiment_id').brier.mean()) & (u.groupby('experiment_id').cal_ece.mean() < u.groupby('experiment_id').ece.mean())).sum())
    return html.Div([kpis(kpi(T('CONFIGURACIONES MEJORADAS'), 80, T('de 96 con probabilidad nativa')), kpi(T('MÉTODO'), 'Platt', T('sigmoide dentro del entrenamiento')),
                          kpi(T('INTERVALOS ECE'), 10, T('uniformes')), kpi('SVM', T('solo sigmoide'), T('sin probabilidad nativa'))),
                     grid(chart(T('Puntaje de Brier por familia'), fb, T('Menor es mejor. Promedio sobre balanceos, optimizadores y folds externos.'), 360),
                          chart(T('ECE por familia'), fe, T('Distancia entre la probabilidad predicha y la frecuencia observada; la rama sigmoide la reduce en casi todas las familias.'), 360)),
                     note(T('1.8 Síntesis de la calibración'), T('La calibración sigmoide redujo a la vez Brier y ECE en 80 de 96 configuraciones con probabilidad nativa. No mejora todas, y no se usó para elegir el ranking principal.'))])


def estadistica():
    st = a.DATA / 'estadistica'; fr = json.loads((st / 'friedman.json').read_text(encoding='utf-8')); nem = json.loads((st / 'nemenyi_estado.json').read_text(encoding='utf-8'))
    r = pd.read_csv(st / 'rangos_medios.csv').sort_values('mean_rank'); cd = nem['critical_difference']
    r['pos'] = range(1, len(r) + 1); best = r.mean_rank.iloc[0]
    inside = r.mean_rank <= best + cd
    fn = go.Figure(go.Scatter(x=r.mean_rank, y=r.pos, mode='markers', marker={'color': [HOT if i == 0 else (BASE if k else GREY) for i, k in enumerate(inside)], 'size': 8},
                              text=r.experiment_id.map(config_name), hovertemplate='%{text}<br>' + T('Rango medio') + ': %{x:.1f}<extra></extra>'))
    fn.add_shape(type='rect', x0=best, x1=best + cd, y0=0, y1=len(r) + 1, fillcolor='rgba(46,134,171,.08)', line_width=0)
    fn.add_annotation(x=best + cd / 2, y=len(r) - 2, text=T('diferencia crítica = {cd}').format(cd=f'{cd:.1f}'), showarrow=False)
    fn.update_yaxes(autorange='reversed', title=T('Posición'), showticklabels=False); fn.update_xaxes(title=T('Rango medio (menor es mejor)'))
    ic = pd.read_csv(st / 'ic_test_suplementarios.csv'); mt = pd.read_csv(st / 'metricas_test_suplementarias.csv')
    rows = []
    for e in a.FINALISTS:
        for m, col in [('F1', 'f1'), ('Recall', 'recall'), ('AUC', 'auc')]:
            x = ic[(ic.experiment_id == e) & (ic.metric == col)].iloc[0]
            val = {'f1': .627, 'recall': .735, 'auc': .844}[col] if e == a.FINALISTS[0] else {'xgboost__smote__random': {'f1': .637, 'recall': .765, 'auc': .842}, 'logistic__smote__grid': {'f1': .625, 'recall': .794, 'auc': .842}}[e][col]
            rows.append({'m': T(a.FINALIST_NAMES[e]), 'metric': m, 'v': val, 'lo': x.low, 'hi': x.high})
    q = pd.DataFrame(rows)
    fig_ic = make_subplots(rows=1, cols=3, subplot_titles=['F1', 'Recall', 'AUC-ROC'], horizontal_spacing=.08)
    for j, m in enumerate(['F1', 'Recall', 'AUC'], 1):
        s = q[q.metric == m]
        fig_ic.add_trace(go.Scatter(x=s.v, y=s.m, mode='markers', marker={'size': 11, 'color': highlight(s.v)}, error_x={'type': 'data', 'symmetric': False, 'array': s.hi - s.v, 'arrayminus': s.v - s.lo, 'color': GREY},
                                    showlegend=False), row=1, col=j)
        if j > 1: fig_ic.update_yaxes(showticklabels=False, row=1, col=j)
    dl = pd.read_csv(st / 'delong_holm.csv')
    dl = pd.DataFrame({T('Modelo A'): [T(a.FINALIST_NAMES.get(x, x)) for x in dl.a], T('Modelo B'): [T(a.FINALIST_NAMES.get(x, x)) for x in dl.b],
                       T('Δ AUC'): dl.delta_auc.round(4), 'z': dl.z.round(3), T('p sin ajustar'): dl.p_raw.round(4),
                       T('IC 95 % Δ AUC'): [f'[{l:.4f}, {h:.4f}]' for l, h in zip(dl.normal_ci_low, dl.normal_ci_high)], T('p Holm'): dl.p_holm.round(3)})
    curves = []
    for e in a.FINALISTS:
        p = a.predictions(e); fpr, tpr, _ = roc_curve(p.y_true, p.score)
        curves.append((f'{T(a.FINALIST_NAMES[e])} · AUC {roc_auc_score(p.y_true, p.score):.3f}', fpr, tpr, HOT if e == a.FINALISTS[0] else (BASE if e == a.FINALISTS[1] else GREY)))
    return [kpis(kpi('FRIEDMAN χ²', round(fr['statistic'], 2), f"p = {fr['p_exploratory']:.1e}", '', 2), kpi('KENDALL W', round(fr['kendall_W'], 2), T('concordancia entre folds'), '', 2),
                 kpi(T('PARES NEMENYI < 0.05'), 31, T('de 6,216 · exploratorio')), kpi(T('p DELONG (HOLM)'), 1.0, T('en los tres pares de finalistas'), '', 1)),
            chart(T('Nemenyi y diferencia crítica'), fn, T('Cada punto es una configuración ordenada por su rango medio en los 5 folds. Las que caen dentro de la banda no se distinguen de la mejor (resaltada).'), 460, False),
            grid(chart(T('Finalistas en el test histórico · IC 95 % bootstrap'), fig_ic, T('Los intervalos de los tres finalistas se superponen ampliamente.'), 330, False),
                 chart(T('Curvas ROC de los finalistas en el test'), roc_animation(curves), T('Pulsa play para trazarlas. Random Forest (resaltado) tiene el mayor AUC, pero la diferencia no es significativa.'), 330)),
            card(html.H3(T('DeLong dirigido con corrección de Holm'), className='chart-title'), table(dl),
                 html.P(T('Ningún par difiere tras Holm (p = 1.0) y todos los intervalos incluyen el cero. No detectar diferencias no prueba que sean equivalentes.'), className='insight'))]


def feature_values(names, rows):
    """Valores de las variables (en escala original) para colorear el beeswarm de SHAP."""
    r = a.RAW.loc[rows]; out = {}
    for f in names:
        n = f.split('__', 1)[1]
        if n == 'AvgChargePerMonth': v = np.where(r.tenure > 0, r.TotalCharges.fillna(0) / r.tenure.replace(0, np.nan), r.MonthlyCharges)
        elif n in r.columns: v = r[n].fillna(0).to_numpy()
        else:
            var = next(c for c in sorted(a.LABELS, key=len, reverse=True) if n.startswith(c + '_')); lvl = n[len(var) + 1:]
            v = (r[var].astype(str) == lvl).astype(float).to_numpy()
        v = pd.Series(np.asarray(v, dtype=float)); out[f] = ((v.rank(pct=True)) if v.nunique() > 2 else v).to_numpy()
    return out


def interpretabilidad():
    m = {k: T(a.FINALIST_NAMES[k]) for k in a.FINALISTS[:2]}
    return [card(field(T('Modelo explicado'), dropdown('explain-model', a.FINALISTS[:2], a.FINALISTS[0], m)), html.Div(id='explain-output'), cls='controls-card'),
            note(T('Explicar no es causar'), T('Las atribuciones describen cómo decide el modelo, no qué provoca el abandono. Los casos locales se eligieron por máxima confianza, no por ser representativos.'))]


# ================================================================ 6. CONCLUSIONES
def conclusiones(d):
    finds = [('layers', N('El perfil de cuenta delata el riesgo'), N('Contrato, antigüedad y servicio de internet concentran la señal, en el EDA y en SHAP.')),
             ('link', N('Empate en la cima'), N('Random Forest y XGBoost con SMOTE (F1 0.638) y la regresión logística (0.634) no se separan estadísticamente.')),
             ('scale', N('Balancear importa más que buscar'), N('SMOTE mejoró el F1 en 6 de 7 familias; cambiar de optimizador casi no movió el resultado.')),
             ('gauge', N('Probabilidades más honestas'), N('La calibración sigmoide mejoró Brier y ECE en 80 de 96 configuraciones.'))]
    limits = [N('Un único conjunto de datos: los folds no son bloques independientes.'), N('El test ya se había usado: es evidencia suplementaria.'),
              N('Los datos son una foto: no garantizan anticipar el abandono futuro.'), N('Sin costos de intervención, el F1 no se traduce en dinero.')]
    steps = [(N('Validación temporal'), N('Evaluar en clientes futuros.')), (N('Costos explícitos'), N('Elegir el umbral con el costo de cada error.')),
             (N('Modelo propuesto'), N('Superar a las referencias en desempeño y tiempo.'))]
    return html.Div([page_head('conclusiones', T('Qué aprendimos, qué sigue siendo incierto y qué falta para llevarlo a una campaña de retención.')),
        subhead(T('Hallazgos principales')),
        html.Div([html.Div([html.Span(icon(i), className='drv-icon'), html.H3(T(t)), html.P(T(s))], className='finding') for i, t, s in finds], className='findings'),
        grid(card(eyebrow(T('LIMITACIONES')), html.Ul([html.Li(T(x)) for x in limits])),
             card(eyebrow(T('PRÓXIMOS PASOS')), html.Div([html.Div([html.Span(f'0{k + 1}', className='obj-n'), html.Div([html.H4(T(t)), html.P(T(s))])], className='next') for k, (t, s) in enumerate(steps)]))),
        card(html.Div([html.Div([eyebrow(T('PAPER DEL PROYECTO')), html.H2(T('Modelo de clasificación binaria para la predicción de abandono de clientes')),
                                 html.P(T('Metodología completa, tablas, figuras y referencias en formato Springer Nature.'))]),
                       html.Div([html.A([icon('file'), T('Leer el paper')], href=BOOK + 'paper.html', target='_blank', className='btn primary'),
                                 html.A([icon('download'), 'PDF'], href=BOOK + 'paper.pdf', target='_blank', className='btn ghost')], className='cover-cta')], className='paper-card'), cls='glow')])


PAY = list(a.RAW.PaymentMethod.unique())
GAUGE_M = {'l': 34, 'r': 34, 't': 50, 'b': 10}


def simulador(d):
    perfil = [grid(card(html.H3(T('Arma un cliente')), field(T('Contrato'), dropdown('sim-contract', ['Month-to-month', 'One year', 'Two year'], 'Month-to-month')),
                        field(T('Internet'), dropdown('sim-internet', ['Fiber optic', 'DSL', 'No'], 'Fiber optic')), field(T('Método de pago'), dropdown('sim-payment', PAY, 'Electronic check')),
                        field(T('Antigüedad (meses)'), dcc.Slider(0, 72, 1, value=4, marks={0: '0', 24: '24', 48: '48', 72: '72'}, id='sim-tenure', tooltip={'placement': 'bottom'})),
                        field(T('Cargo mensual ($)'), dcc.Slider(18, 119, 1, value=85, marks={20: '20', 60: '60', 100: '100'}, id='sim-monthly', tooltip={'placement': 'bottom'})), cls='controls-card'),
                   html.Div(id='sim-output')),
              note(T('¿De dónde sale este riesgo?'), T('Es la tasa de abandono observada en clientes de entrenamiento con un perfil parecido, no una predicción del modelo. '
                   'Sirve para explorar patrones; para la predicción del modelo usa la pestaña de clientes reales.'))]
    reales = [card(html.Div([field(T('Cliente del test histórico'), dcc.Dropdown([{'label': a.RAW.customerID[i], 'value': i} for i in a.TEST_IDS], int(a.TEST_IDS[0]), id='real-id', clearable=False)),
                             html.Button([icon('shuffle'), T('Cliente al azar')], id='real-random', className='btn ghost', n_clicks=0)], className='controls'),
                    html.Div(id='real-output'), cls='controls-card')]
    umbral = [card(html.Div([field(T('Modelo finalista'), dropdown('threshold-model', a.FINALISTS, a.FINALISTS[0], {k: T(v) for k, v in a.FINALIST_NAMES.items()})),
                             field(T('Umbral de alerta'), dcc.Slider(.05, .95, .01, value=.5, marks={.1: '0.10', .3: '0.30', .5: '0.50', .7: '0.70', .9: '0.90'}, id='threshold', tooltip={'always_visible': True}))],
                            className='controls'), html.Div(id='threshold-output'), cls='controls-card'),
              note(T('Decisión de negocio'), T('Bajar el umbral detecta más clientes en riesgo pero dispara más falsas alertas. Elegirlo bien requiere conocer el costo de cada contacto y de cada cliente perdido.'))]
    return html.Div([page_head('simulador', T('Juega con perfiles, revisa clientes reales y decide dónde poner la alarma de retención.')),
                     note(T('Herramienta creada para el dash'), T('Combina los datos de entrenamiento y las predicciones guardadas de los modelos finalistas para explorar casos; no corresponde a un capítulo del libro.')),
                     tabs('sim-tabs', [(T('Perfil de riesgo'), perfil), (T('Clientes reales'), reales), (T('Umbral de alerta'), umbral)])])


def gauge(value, title, color):
    return go.Figure(go.Indicator(mode='gauge+number', value=value * 100, number={'suffix': ' %', 'font': {'size': 34}}, title={'text': title, 'font': {'size': 13}},
                                  gauge={'axis': {'range': [0, 100], 'tickcolor': '#8792a8'}, 'bar': {'color': color, 'thickness': .28}, 'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                                         'steps': [{'range': [0, 35], 'color': 'rgba(59,181,122,.14)'}, {'range': [35, 60], 'color': 'rgba(242,169,59,.14)'}, {'range': [60, 100], 'color': 'rgba(232,86,108,.16)'}]}))


def level(p): return ('alto', HIGH) if p >= .6 else ('medio', MID) if p >= .35 else ('bajo', LOW)


def signal(p):
    lit = max(1, round((1 - p) * 5))
    return html.Div([html.B(className='lit' if i < lit else '', style={'height': f'{12 + i * 9}px'}) for i in range(5)], className='signal ' + level(p)[0])


def autoras(d):
    people = [('laura.jpg', 'Laura S. Rivera Figueredo', 'sriveral@uninorte.edu.co', 'https://github.com/lauraformore', 'https://www.linkedin.com/in/laura-s-rivera-figueredo-862451315/'),
              ('nataly.jpg', 'Natalý A. Cárdenas Izaquita', 'nizaquita@uninorte.edu.co', 'https://github.com/Nataly-Cardenas', 'https://www.linkedin.com/in/n%C3%A1taly-c%C3%A1rdenas-iz%C3%A1quita')]
    res = [('code', T('Código y notebooks'), T('Repositorio Telco_Churn_Entregable'), REPO), ('sliders', T('Este dashboard'), T('Repositorio Dash_Telco_Churn'), DASH_REPO),
           ('table', T('Conjunto de datos'), 'Kaggle · Telco Customer Churn (IBM)', 'https://www.kaggle.com/datasets/blastchar/telco-customer-churn'), ('file', 'Paper', T('Versión web y PDF'), BOOK + 'paper.html')]
    return html.Div([page_head('autoras', T('El equipo detrás del proyecto y los recursos para profundizar.')),
        html.Div([html.Div([html.Div(html.Img(src=app.get_asset_url('img/' + photo), alt=T('Foto de {n}').format(n=n)), className='avatar'), html.H2(n),
                            html.P(T('Ciencia de Datos · Universidad del Norte'), className='muted'),
                            html.Div([html.A([icon('linkedin', 'sm'), 'LinkedIn'], href=li, target='_blank', className='chip-link linkedin'),
                                      html.A([icon('code', 'sm'), 'GitHub'], href=g, target='_blank', className='chip-link'),
                                      html.A([icon('mail', 'sm'), m], href='mailto:' + m, className='chip-link')], className='author-links')],
                           className='author') for photo, n, m, g, li in people], className='authors'),
        html.P(T('Departamento de Matemáticas, Física y Ciencia de Datos · Universidad del Norte · Barranquilla, Colombia'), className='center muted'),
        html.A([html.Span(icon('book'), className='book-icon'), html.Div([eyebrow('JUPYTER BOOK'), html.H2(T('Lee el proyecto completo')), html.P(T('Introducción, marco teórico, EDA, preprocesamiento, modelado y paper.'))]),
                html.Span(icon('arrow'), className='book-arrow')], href=BOOK, target='_blank', className='book-cta'),
        grid(*[html.A([html.H4([icon(i), t]), html.P(s)], href=u, target='_blank', className='res') for i, t, s, u in res], cols=4)])


def pretty(feature):
    """cat__Contract_Two year → Contrato: Dos años; num__tenure → Antigüedad (meses)."""
    name = feature.split('__', 1)[-1]
    for var in sorted(a.LABELS, key=len, reverse=True):
        if name == var: return label(var)
        if name.startswith(var + '_'): return f'{label(var)}: {cat(name[len(var) + 1:])}'
    return name


PAGES = {'inicio': lambda d: cover(), 'introduccion': introduccion, 'eda': eda_page, 'preprocesamiento': preprocesamiento, 'modelado': modelado,
         'simulador': simulador, 'conclusiones': conclusiones, 'autoras': autoras}

# ---------------------------------------------------------------- layout
TICKER = [N('Contrato mes a mes: 42.7 % de abandono'), N('Fibra óptica: 42.1 %'), N('Cheque electrónico: 45.7 %'), N('Primer año: el periodo más frágil'),
          N('Sin seguridad en línea: 41.9 %'), N('Mejor F1: 0.638 · Random Forest + SMOTE'), N('SMOTE mejoró 6 de 7 familias'), N('Contrato, internet y antigüedad lideran SHAP')]


def ticker():
    return html.Div([html.Span([html.I(className='tick-dot'), T(t)]) for t in TICKER * 2], className='ticker-track')


def split_options(): return options(['train', 'test', 'all'], {'train': T('Entrenamiento · 5,634'), 'test': T('Test · 1,409'), 'all': T('Todo · 7,043')})
def contract_options(): return options(['Todos'] + sorted(a.RAW.Contract.unique()), {'Todos': T('Todos')})
def internet_options(): return options(['Todos'] + sorted(a.RAW.InternetService.unique()), {'Todos': T('Todos')})


app.layout = html.Div([
    dcc.Location(id='url'), dcc.Store(id='lang', storage_type='local', data='es'), dcc.Store(id='theme', storage_type='local', data='dark'),
    html.Header([
        dcc.Link([html.Span([html.B(), html.B(), html.B(), html.B()], className='logo-sig'), html.Span(id='logo-text', className='logo-text')], href='#inicio', className='logo'),
        html.Nav([dcc.Link([icon(i, 'nav-i'), html.Span(id='nav-t-' + k)], href='#' + k, id='nav-' + k, className='nav-link') for k, _, i in NAV], className='nav'),
        html.Div([html.Div([html.Button('ES', id='lang-es', className='seg', n_clicks=0), html.Button('EN', id='lang-en', className='seg', n_clicks=0)], className='segmented', id='lang-switch'),
                  html.Button([icon('sun', 'sun'), icon('moon', 'moon')], id='theme-btn', className='icon-btn', n_clicks=0),
                  html.A([icon('book', 'sm'), html.Span('Jupyter Book', className='book-btn-t')], href=BOOK, target='_blank', className='btn small')], className='top-actions')], className='topbar', id='topbar'),
    html.Div(id='ticker', className='ticker'),
    html.Div([field('', dropdown('split', ['train', 'test', 'all'], 'train'), id='f-split'),
              field('', dropdown('contract', ['Todos'] + sorted(a.RAW.Contract.unique()), 'Todos'), id='f-contract'),
              field('', dropdown('internet', ['Todos'] + sorted(a.RAW.InternetService.unique()), 'Todos'), id='f-internet'),
              field('', dcc.RangeSlider(0, 72, 1, value=[0, 72], marks={0: '0', 24: '24', 48: '48', 72: '72'}, id='tenure', tooltip={'placement': 'bottom'}), id='f-tenure'),
              html.Button([icon('reset', 'sm'), html.Span(id='reset-t')], id='reset', className='btn ghost small')], className='filters', id='filters'),
    html.Main(id='content'),
    html.Footer(id='footer', className='footer'),
    chatbot.layout(),
    dcc.Download(id='download')], id='shell')


@app.callback(Output('logo-text', 'children'), *[Output('nav-t-' + k, 'children') for k, _, _ in NAV], Output('ticker', 'children'),
              Output('f-split', 'children'), Output('f-contract', 'children'), Output('f-internet', 'children'), Output('f-tenure', 'children'), Output('reset-t', 'children'),
              Output('split', 'options'), Output('contract', 'options'), Output('internet', 'options'), Output('footer', 'children'),
              Output('lang-es', 'className'), Output('lang-en', 'className'), Output('theme-btn', 'title'), Input('lang', 'data'))
def chrome(lang):
    """Textos fijos de la interfaz: se recalculan solo al cambiar de idioma."""
    set_lang(lang)
    footer = [html.Span(T('Señal perdida · Telco Customer Churn')), html.Span('Laura Rivera & Natalý Cárdenas · Universidad del Norte')]
    return ([html.B(T('Señal')), ' ' + T('perdida')], *[T(t) for _, t, _ in NAV], ticker(),
            T('Universo'), T('Contrato'), T('Internet'), T('Antigüedad (meses)'), T('Restablecer'), split_options(), contract_options(), internet_options(), footer,
            'seg on' if lang != 'en' else 'seg', 'seg on' if lang == 'en' else 'seg', T('Cambiar tema claro / oscuro'))


@app.callback(Output('lang', 'data'), Input('lang-es', 'n_clicks'), Input('lang-en', 'n_clicks'), prevent_initial_call=True)
def choose_lang(_es, _en): return 'en' if ctx.triggered_id == 'lang-en' else 'es'


@app.callback(Output('content', 'children'), *[Output('nav-' + k, 'className') for k, _, _ in NAV], Output('filters', 'style'), Output('shell', 'className'),
              Input('url', 'hash'), Input('lang', 'data'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'))
def route(hash_, lang, split, contract, internet, tenure):
    set_lang(lang)
    key = (hash_ or '#inicio').lstrip('#')
    if key not in PAGES: key = 'inicio'
    if ctx.triggered_id in {'split', 'contract', 'internet', 'tenure'} and key not in DATA_PAGES:
        return (no_update,) * (len(NAV) + 3)
    d = a.cohort(split, contract, internet, tenure)
    return (html.Div(PAGES[key](d), className='page ' + key), *['nav-link on' if key == k else 'nav-link' for k, _, _ in NAV],
            {} if key in DATA_PAGES else {'display': 'none'}, 'shell cover-mode' if key == 'inicio' else 'shell')


@app.callback(Output('split', 'value'), Output('contract', 'value'), Output('internet', 'value'), Output('tenure', 'value'), Input('reset', 'n_clicks'), prevent_initial_call=True)
def reset(_): return 'train', 'Todos', 'Todos', [0, 72]


FILTERS = [Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value')]


# ---------------------------------------------------------------- EDA: univariado
@app.callback(Output('uni-num-output', 'children'), Input('uni-num', 'value'), *FILTERS, State('lang', 'data'))
def uni_num(var, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    s = d[var].dropna()
    if s.empty: return empty()
    f = make_subplots(rows=1, cols=2, column_widths=[.7, .3], subplot_titles=[T('Distribución'), T('Rango y cuartiles')])
    f.add_trace(go.Histogram(x=s, nbinsx=30, marker_color=BASE, name=T('Frecuencia')), row=1, col=1)
    f.add_trace(go.Box(y=s, name=var, marker_color=BASE, boxpoints='outliers', quartilemethod='linear'), row=1, col=2)
    f.add_vline(x=s.mean(), line_dash='dash', line_color=HOT, row=1, col=1, annotation_text=T('media {m}').format(m=f'{s.mean():.1f}'))
    f.update_xaxes(title_text=var, row=1, col=1); f.update_yaxes(title_text=T('Clientes'), row=1, col=1)
    return html.Div([graph(f, 380, False), html.P(T('{n} valores · media {m} · mediana {md} · asimetría {s}.').format(n=f'{len(s):,}', m=f'{s.mean():.2f}', md=f'{s.median():.2f}', s=f'{s.skew():.2f}'), className='insight')])


@app.callback(Output('uni-cat-output', 'children'), Input('uni-cat', 'value'), *FILTERS, State('lang', 'data'))
def uni_cat(var, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    variables = a.CATEGORICAL if var == '__all__' else [var]
    cols = 4 if len(variables) > 1 else 1; rows = int(np.ceil(len(variables) / cols))
    f = make_subplots(rows=rows, cols=cols, subplot_titles=variables, horizontal_spacing=.12, vertical_spacing=.08 if rows > 1 else .1)
    for i, c in enumerate(variables):
        v = d[c].map(cat).astype(str).value_counts().sort_values()
        pct = 100 * v / v.sum()
        f.add_trace(go.Bar(x=v.values, y=v.index, orientation='h', marker_color=BASE, text=pct.map('{:.1f}%'.format), textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}<br>' + T('Clientes') + ': %{x}<extra></extra>'), row=i // cols + 1, col=i % cols + 1)
    h = 320 if len(variables) == 1 else 230 * rows
    insight = T('Distribución de todas las variables categóricas en la selección.') if len(variables) > 1 else T('Frecuencia de cada categoría de {v}.').format(v=var)
    return html.Div([graph(f, h, False), html.P(insight, className='insight')])


# ---------------------------------------------------------------- EDA: bivariado
@app.callback(Output('bi-num-output', 'children'), Output('bi-pb-output', 'children'), Input('bi-num', 'value'), *FILTERS, State('lang', 'data'))
def bi_num(var, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if d.empty or d.Churn.nunique() < 2: return empty(), None
    x = d.assign(Estado=d.Churn.map(churn_label))
    f = px.box(x, x='Estado', y=var, color='Estado', color_discrete_map=outcome_colors(), points='outliers', category_orders={'Estado': [T('Permanece'), T('Abandona')]}, labels={'Estado': ''})
    f.update_traces(quartilemethod='linear')
    num, _, _ = eda.holm_family(d)
    by = x.groupby('Estado')[var].agg(['count', 'mean', 'std', 'median']).reset_index().rename(columns={'count': 'N', 'mean': 'Media', 'std': 'DS', 'median': 'Mediana'})
    test = num[num.Variable == var]
    rk = num.assign(abs_r=num['r punto-biserial'].abs()).sort_values('abs_r')
    fp = go.Figure(go.Bar(x=rk['r punto-biserial'], y=rk.Variable, orientation='h', marker_color=highlight(rk.abs_r), text=rk['r punto-biserial'].map('{:+.3f}'.format), textposition='outside'))
    fp.add_vline(x=0, line_color=GREY); fp.update_xaxes(title=T('r punto-biserial con el abandono'), range=[-.45, .3])
    left = html.Div([grid(graph(f, 380, False), html.Div([html.H3(T('Resumen por clase'), className='chart-title'), table(tt(by)), html.H3(T('Prueba estadística'), className='chart-title'),
                                                          table(tt(test.drop(columns=['Variable'])))])),
                     html.P(T('Mann–Whitney U para tenure y TotalCharges, t de Welch para MonthlyCharges; p corregido con Holm sobre los 19 contrastes.'), className='insight')])
    return left, chart(T('Asociación punto-biserial'), fp, T('Correlación de cada variable numérica con el abandono; tenure (resaltada) es la más fuerte.'), 280, False)


@app.callback(Output('bi-cat-output', 'children'), Output('bi-holm-output', 'children'), Input('bi-cat', 'value'), *FILTERS, State('lang', 'data'))
def bi_cat(var, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if d.empty or d.Churn.nunique() < 2: return empty(), None
    ct = pd.crosstab(d[var], d.Churn).reindex(columns=['No', 'Yes'], fill_value=0)
    t = ct.rename_axis('c').reset_index().melt(id_vars='c', var_name='Churn', value_name='n'); t['c'] = t.c.map(cat).astype(str); t['Churn'] = t.Churn.map(churn_label)
    f = px.bar(t, x='n', y='c', color='Churn', barmode='group', orientation='h', color_discrete_map=outcome_colors(), category_orders={'Churn': [T('Permanece'), T('Abandona')]},
               labels={'n': T('Clientes (frecuencia absoluta)'), 'c': ''}, text='n')
    rate = (100 * ct.Yes / ct.sum(axis=1)).rename(index=cat)
    _, catt, fam = eda.holm_family(d)
    row = catt[catt.Variable == var].drop(columns=['Variable'])
    left = html.Div([grid(graph(f, max(300, 90 + 70 * len(ct)), True),
                          html.Div([html.H3(T('Tasa de abandono por categoría'), className='chart-title'),
                                    html.Div([html.Div([html.Span(str(k)), html.Div(html.I(style={'width': f'{v:.1f}%'}, className='hot' if v == rate.max() else ''), className='rate-bar'), html.B(f'{v:.1f}%')], className='rate-row')
                                              for k, v in rate.items()], className='rates'),
                                    html.H3(T('Chi-cuadrado'), className='chart-title'), table(tt(row))]))])
    v = catt.sort_values('V de Cramér')
    fv = go.Figure(go.Bar(x=v['V de Cramér'], y=v.Variable, orientation='h', marker_color=highlight(v['V de Cramér']), text=v['V de Cramér'].map('{:.3f}'.format), textposition='outside',
                          customdata=v['p Holm'], hovertemplate='%{y}<br>V = %{x:.3f}<br>p Holm = %{customdata:.2e}<extra></extra>'))
    fv.update_xaxes(title=T('V de Cramér con el abandono'), range=[0, .5])
    holm = fam[['Variable', 'p', 'p Holm', 'Rechaza H0']].rename(columns={'p': 'p original'})
    right = html.Div([chart(T('Fuerza de la asociación de cada variable categórica'), fv, T('Contrato (resaltada) es la más asociada; gender y PhoneService no rechazan H0 tras Holm.'), 520, False),
                      details([icon('table'), T('Control de comparaciones múltiples (Holm)')], table(tt(holm), page_size=19))])
    return left, right


# ---------------------------------------------------------------- EDA: multivariado y faltantes
@app.callback(Output('multi-output', 'children'), *FILTERS, State('lang', 'data'))
def multi(split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if len(d) < 30: return empty()
    fq = make_subplots(rows=1, cols=3, subplot_titles=a.NUMERIC, horizontal_spacing=.08)
    for j, c in enumerate(a.NUMERIC, 1):
        theo, obs, slope, inter = eda.qq(d[c]); k = np.linspace(0, len(theo) - 1, min(600, len(theo))).astype(int)
        fq.add_trace(go.Scatter(x=theo[k], y=obs[k], mode='markers', marker={'size': 4, 'color': BASE, 'opacity': .55}, showlegend=False), row=1, col=j)
        fq.add_trace(go.Scatter(x=[theo.min(), theo.max()], y=[inter + slope * theo.min(), inter + slope * theo.max()], mode='lines', line={'color': HOT, 'width': 2}, showlegend=False), row=1, col=j)
        fq.update_xaxes(title_text=T('Cuantiles teóricos'), row=1, col=j)
    fq.update_yaxes(title_text=T('Cuantiles observados'), row=1, col=1)
    corr = d[a.NUMERIC].corr(method='spearman')
    fh = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale=DIVERGING, text_auto='.3f', aspect='auto')
    sp = d.sample(min(1500, len(d)), random_state=42).assign(Estado=lambda x: x.Churn.map(churn_label))
    fs = px.scatter_matrix(sp, dimensions=a.NUMERIC, color='Estado', color_discrete_map=outcome_colors(), opacity=.35, category_orders={'Estado': [T('Permanece'), T('Abandona')]})
    fs.update_traces(diagonal_visible=False, showupperhalf=False, marker_size=3)
    fk = make_subplots(rows=1, cols=2, subplot_titles=['MonthlyCharges', 'TotalCharges'])
    for j, c in enumerate(['MonthlyCharges', 'TotalCharges'], 1):
        g, dens = eda.kde(d, c)
        for k, y in dens.items():
            fk.add_trace(go.Scatter(x=g, y=y, mode='lines', name=churn_label(k), line={'color': BASE if k == 'No' else HOT, 'width': 2.5}, fill='tozeroy', showlegend=j == 1), row=1, col=j)
    fk.update_yaxes(title_text=T('Densidad estimada'), row=1, col=1)
    fc = px.scatter(sp, x='tenure', y='MonthlyCharges', color='Estado', facet_col='Contract', color_discrete_map=outcome_colors(), opacity=.4, render_mode='webgl',
                    category_orders={'Contract': ['Month-to-month', 'One year', 'Two year'], 'Estado': [T('Permanece'), T('Abandona')]})
    fc.update_traces(marker_size=4); fc.for_each_annotation(lambda x: x.update(text=cat(x.text.split('=')[-1])))
    return html.Div([chart(T('Prueba de normalidad · gráficos Q-Q'), fq, T('Si los puntos siguieran la línea resaltada la variable sería normal; ninguna lo es, por eso se usa Spearman.'), 360, False),
                     card(table(tt(eda.normality(d)))),
                     grid(chart(T('Matriz de correlaciones de Spearman'), fh, T('tenure y TotalCharges comparten mucha información (ρ ≈ 0.89).'), 380, False),
                          card(html.H3(T('Asociación entre variables categóricas'), className='chart-title'), table(tt(eda.categorical_pairs(d))),
                               html.H3(T('Identificación de posibles redundancias'), className='chart-title'), table(tt(eda.redundancy(d))))),
                     subhead(T('Visualizaciones avanzadas')),
                     chart(T('Matriz de dispersión por clase'), fs, T('Muestra de hasta 1,500 clientes. Los que abandonan se concentran en baja antigüedad y cargos mensuales altos.'), 560),
                     chart(T('Densidad por clase'), fk, T('Estimación de densidad de los cargos para quienes permanecen y quienes abandonan.'), 360),
                     chart(T('Antigüedad y cargo mensual por contrato'), fc, T('El abandono se concentra en contratos mes a mes con poca antigüedad.'), 400)])


@app.callback(Output('missing-output', 'children'), *FILTERS, State('lang', 'data'))
def missing(split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    mt = eda.missing_table(d)
    pat = d[['TotalCharges', 'tenure']].assign(miss=d.TotalCharges.isna().astype(int)).sort_values('miss', ascending=False)
    fp = px.imshow(pat[['miss']].to_numpy().T, color_continuous_scale=[[0, 'rgba(46,134,171,.15)'], [1, HOT]], zmin=0, zmax=1, aspect='auto', labels={'x': T('Posición ordenada en la selección'), 'color': T('Ausente')})
    fp.update_yaxes(tickvals=[0], ticktext=['TotalCharges']); fp.update_coloraxes(showscale=False)
    absent = d.loc[d.TotalCharges.isna(), ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    rows = d[a.RAW.columns[:21]].isna().sum(axis=1).value_counts().sort_index().rename_axis('Nulos por fila').reset_index(name='Filas')
    return html.Div([kpis(kpi(T('VALORES FALTANTES'), int(mt.Nulos.sum()) if not mt.empty else 0, T('solo en TotalCharges')), kpi(T('PORCENTAJE'), round(100 * d.TotalCharges.isna().mean(), 2), T('de la selección'), '', 2, ' %'),
                          kpi(T('CON TENURE = 0'), int(d.loc[d.TotalCharges.isna(), 'tenure'].eq(0).sum()), T('clientes recién dados de alta')), kpi(T('FILAS AFECTADAS'), int(d.TotalCharges.isna().sum()), T('un faltante cada una'))),
                     chart(T('Patrón de valores faltantes'), fp, T('Cada línea vertical es un cliente; los faltantes (resaltados) son pocos y se concentran en clientes con tenure 0.'), 220, False),
                     grid(card(html.H3(T('Clientes con TotalCharges ausente'), className='chart-title'), table(tt(absent))),
                          card(html.H3(T('Relación con la variable objetivo y evaluación de MCAR'), className='chart-title'), table(tt(eda.mcar_tests(d))), table(tt(rows)),
                               html.P(T('La ausencia no se asocia con el abandono, pero coincide por completo con tenure = 0: no es MCAR, es estructural, y se imputa con 0.'), className='insight')))])


# ---------------------------------------------------------------- modelado
@app.callback(Output('res-output', 'children'), Input('res-family', 'value'), State('lang', 'data'))
def res_output(family, lang='es'):
    set_lang(lang); e = BEST_BY_FAMILY[family]; o = oof()[lambda x: x.experiment == e]
    cm = confusion_matrix(o.y_true, o.y_pred)
    fcm = px.imshow(cm, x=[T('Predice permanencia'), T('Predice abandono')], y=[T('Permanece'), T('Abandona')], text_auto=True, color_continuous_scale=SEQ, aspect='auto')
    fcm.update_coloraxes(showscale=False); fcm.update_traces(textfont_size=17)
    fpr, tpr, _ = roc_curve(o.y_true, o.score_positive); auc = roc_auc_score(o.y_true, o.score_positive)
    froc = go.Figure([go.Scatter(x=fpr, y=tpr, mode='lines', line={'color': BASE, 'width': 2.6}, name=f'AUC OOF = {auc:.3f}'),
                      go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line={'dash': 'dash', 'color': GREY}, name=T('Azar'))])
    froc.update_xaxes(title=T('Tasa de falsos positivos')); froc.update_yaxes(title='Recall')
    fr = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line={'dash': 'dash', 'color': GREY}, name=T('Calibración perfecta')))
    for col, name, color in [('probability_native', T('Nativa'), GREY), ('probability_sigmoid', T('Sigmoide'), BASE)]:
        if o[col].notna().any():
            b = pd.cut(o[col], np.linspace(0, 1, 11), include_lowest=True); g = o.groupby(b, observed=True).agg(x=(col, 'mean'), y=('y_true', 'mean')).dropna()
            fr.add_trace(go.Scatter(x=g.x, y=g.y, mode='lines+markers', name=name, line={'color': color, 'width': 2.4}))
    fr.update_xaxes(title=T('Probabilidad media'), range=[0, 1]); fr.update_yaxes(title=T('Frecuencia de abandono'), range=[0, 1])
    row = a.SUMMARY.set_index('experiment_id').loc[e]
    m = [kpi(T(METRICS[k]).upper() if k == 'f1_positive' else T(METRICS[k]).upper(), round(row[k + '_mean'], 3), '± ' + f"{row[k + '_std']:.3f}", 'hot' if k == 'f1_positive' else '', 3) for k in METRICS]
    return html.Div([eyebrow(config_name(e)), kpis(*m, cls='k5'),
                     html.Div([chart(T('Matriz de confusión OOF'), fcm, T('Predicciones fuera de muestra de los 5 folds externos ({n} clientes).').format(n=f'{len(o):,}'), 330, False),
                               chart(T('Curva ROC OOF'), froc, '', 330), chart(T('Confiabilidad · 10 intervalos'), fr, T('Más cerca de la diagonal = probabilidades mejor calibradas.'), 330)], className='grid g3')])


@app.callback(Output('models-output', 'children'), Input('model-family', 'value'), Input('model-balance', 'value'), Input('model-metric', 'value'), State('lang', 'data'))
def model_output(family, balance, metric, lang='es'):
    set_lang(lang); d = a.SUMMARY.copy()
    if family != 'Todas': d = d[d.model == family]
    if balance != 'Todos': d = d[d.balance == balance]
    d = d.sort_values(metric + '_mean', ascending=False); top = d.head(12).iloc[::-1].copy()
    top['n'] = top.experiment_id.map(config_name)
    f = go.Figure(go.Bar(x=top[metric + '_mean'], y=top.n, orientation='h', error_x={'type': 'data', 'array': top[metric + '_std'], 'color': GREY}, marker_color=highlight(top[metric + '_mean']),
                         text=top[metric + '_mean'].map('{:.3f}'.format), textposition='inside'))
    lo = max(0, top[metric + '_mean'].min() - .06)
    f.update_xaxes(title=T(METRICS[metric]) + ' · ' + T('media externa'), range=[lo, min(1, top[metric + '_mean'].max() + .04)])
    heat = px.imshow(d.pivot_table(index='model', columns='balance', values=metric + '_mean', aggfunc='mean').rename(index=model_name, columns=balance_name), text_auto='.3f',
                     color_continuous_scale=SEQ, aspect='auto', labels={'x': '', 'y': '', 'color': T(METRICS[metric])})
    view = d[['model', 'balance', 'optimizer', metric + '_mean', metric + '_std', 'recall_positive_mean', 'precision_positive_mean', 'roc_auc_mean']].copy()
    view = view.loc[:, ~view.columns.duplicated()]; view['model'] = view.model.map(model_name); view['balance'] = view.balance.map(balance_name); view['optimizer'] = view.optimizer.map(optimizer_name)
    return html.Div([grid(chart(T('Ranking de configuraciones · top {k} de {n}').format(k=len(top), n=len(d)), f, T('La mejor configuración está resaltada; bigotes: desviación estándar entre folds.'), 480, False),
                          chart(T('Familia × balanceo'), heat, T('Promedio sobre optimizadores: la columna SMOTE es la más alta en casi todas las familias.'), 480, False)),
                     details([icon('table'), T('Tabla completa · {n} configuraciones').format(n=len(d))], table(view, filter_action='native'))])


@app.callback(Output('opt-output', 'children'), Input('opt-family', 'value'), State('lang', 'data'))
def opt_output(family, lang='es'):
    set_lang(lang); d = anytime(); d = d[(d.model == family) & (d.replicate == 0)]
    g = d.groupby(['optimizer', 'evaluation']).agg(f1=('best_inner_f1', 'mean'), sec=('seconds', 'mean')).reset_index().sort_values(['optimizer', 'evaluation'])
    best_opt = g[g.evaluation == g.evaluation.max()].sort_values('f1').optimizer.iloc[-1]
    dashes = {'grid': 'dot', 'random': 'dash', 'bayesian_tpe': 'solid', 'genetic': 'dashdot'}

    def lines(x, xtitle):
        f = go.Figure()
        for o, s in g.groupby('optimizer'):
            f.add_trace(go.Scatter(x=s[x], y=s.f1, mode='lines', name=optimizer_name(o), line={'color': HOT if o == best_opt else BASE, 'dash': dashes[o], 'width': 3 if o == best_opt else 2}))
        f.update_xaxes(title=xtitle); f.update_yaxes(title=T('Mejor F1 interno'))
        return f
    dv = diversity()[lambda x: x.model == family]
    fd = go.Figure([go.Scatter(x=dv.generation, y=dv.unique_fraction, mode='lines+markers', name=T('Fracción única'), line={'color': BASE}),
                    go.Scatter(x=dv.generation, y=dv.mean_normalized_distance, mode='lines+markers', name=T('Distancia normalizada'), line={'color': GREY, 'dash': 'dash'})])
    fd.update_xaxes(title=T('Generación')); fd.update_yaxes(title=T('Diversidad'), range=[0, 1.05])
    u = a.UNITS[(a.UNITS.model == family) & (a.UNITS.replicate == 0)].copy(); u['b'] = u.optimizer.map(optimizer_name)
    fb = px.box(u, x='b', y='search_seconds', labels={'search_seconds': T('Segundos de búsqueda'), 'b': ''}); fb.update_traces(marker_color=BASE)
    return html.Div([grid(chart(T('Mejor F1 interno por evaluaciones únicas'), lines('evaluation', T('Evaluaciones únicas')), T('A igual presupuesto (24 evaluaciones) los buscadores convergen; el mejor al final está resaltado.'), 380),
                          chart(T('Mejor F1 interno por tiempo acumulado'), lines('sec', T('Tiempo acumulado (s)')), T('La misma búsqueda medida en segundos.'), 380)),
                     grid(chart(T('2.1 Diversidad del algoritmo genético'), fd, T('La población pierde diversidad con las generaciones: converge hacia pocas configuraciones.'), 340),
                          chart(T('2.2 Presupuesto y coste empírico'), fb, T('Segundos por unidad externa en el equipo del experimento.'), 340, False))])


@app.callback(Output('explain-output', 'children'), Input('explain-model', 'value'), State('lang', 'data'))
def explain_output(experiment, lang='es'):
    set_lang(lang)
    s = shap_sample()[lambda x: x.model == experiment]
    order = s.groupby('feature').shap.apply(lambda v: v.abs().mean()).sort_values().index.tolist()
    vals = feature_values(order, s.row_id.unique())
    rid = {r: i for i, r in enumerate(s.row_id.unique())}
    s = s.assign(y=s.feature.map({f: i for i, f in enumerate(order)}), c=[vals[f][rid[r]] for f, r in zip(s.feature, s.row_id)])
    jitter = np.random.default_rng(0).normal(0, .12, len(s))
    fb = go.Figure(go.Scattergl(x=s.shap, y=s.y + jitter, mode='markers', marker={'size': 4, 'color': s.c, 'colorscale': [[0, BASE], [1, HOT]], 'opacity': .75,
                                                                                    'colorbar': {'title': T('Valor'), 'tickvals': [0, 1], 'ticktext': [T('bajo'), T('alto')], 'thickness': 10}},
                                hovertemplate='SHAP %{x:.3f}<extra></extra>'))
    fb.update_yaxes(tickvals=list(range(len(order))), ticktext=[pretty(f) for f in order]); fb.update_xaxes(title=T('Valor SHAP (impacto en la probabilidad)'))
    fb.add_vline(x=0, line_color=GREY)
    path = a.DATA / 'interpretabilidad' / experiment
    cases = json.loads((path / 'casos_locales.json').read_text(encoding='utf-8')); lime = json.loads((path / 'lime_resultados.json').read_text(encoding='utf-8'))
    local = []
    for case in cases:
        row = case['row_id']; sh = pd.read_csv(path / f'shap_local_{row}.csv'); sh = sh.reindex(sh.shap.abs().sort_values(ascending=False).index)
        top = sh.head(10); rest = sh.shap.iloc[10:].sum(); base = float(s.base.iloc[0])
        fw = go.Figure(go.Waterfall(orientation='h', y=['E[f(X)]', *[pretty(f) for f in top.feature], T('{n} variables más').format(n=len(sh) - 10), 'f(x)'],
                                    x=[base, *top.shap, rest, 0], measure=['absolute', *['relative'] * (len(top) + 1), 'total'],
                                    increasing={'marker': {'color': HOT}}, decreasing={'marker': {'color': BASE}}, totals={'marker': {'color': GREY}}, connector={'line': {'color': GREY, 'width': 1}}))
        fw.update_yaxes(autorange='reversed'); fw.update_xaxes(title=T('Probabilidad de abandono'))
        lw = next(x for x in lime if x['row_id'] == row and x['seed'] == 42)
        ld = pd.DataFrame(lw['weights'], columns=['r', 'w']).iloc[::-1]
        fl = go.Figure(go.Bar(x=ld.w, y=ld.r, orientation='h', marker_color=[HOT if v > 0 else BASE for v in ld.w])); fl.add_vline(x=0, line_color=GREY)
        fl.update_xaxes(title=T('Peso del sustituto local'))
        ok = case['correct']
        local.append(card(html.Div([html.H3(T('Cliente') + ' ' + a.RAW.customerID[row]), html.Span([icon('check' if ok else 'x', 'sm'), T('acierto') if ok else T('error')], className='pill ' + ('ok' if ok else 'bad'))], className='card-head'),
                          html.P(T('Realidad: {r} · probabilidad del modelo: {p}').format(r=T('abandonó') if case['y_true'] else T('permaneció'), p=f"{case['probability']:.1%}"), className='muted'),
                          grid(html.Div([html.H4('SHAP · waterfall'), graph(fw, 420, False)]), html.Div([html.H4(T('LIME · clase abandono')), graph(fl, 420, False),
                                                                                                         html.P(T('R² ponderado del sustituto: {r}').format(r=f"{lw['weighted_r2']:.2f}"), className='insight')]))))
    return html.Div([chart(T('SHAP global · probabilidad nativa de abandono'), fb, T('Cada punto es un cliente. A la derecha empuja hacia el abandono; el color indica el valor de la variable (resaltado = alto).'), 560, False),
                     subhead(T('Casos locales')), *local])


@app.callback(Output('sim-output', 'children'), Input('sim-contract', 'value'), Input('sim-internet', 'value'), Input('sim-payment', 'value'), Input('sim-tenure', 'value'), Input('sim-monthly', 'value'), State('lang', 'data'))
def sim_output(contract, internet, payment, tenure, monthly, lang='es'):
    set_lang(lang)
    s, w = a.similar_customers(contract, internet, payment, tenure, monthly)
    if s.empty: return card(note(T('Sin clientes parecidos'), T('Prueba con otra combinación de contrato e internet.')))
    r = s.target.mean(); lv, col = level(r)
    window = T('antigüedad ±{t} meses · cargo ±${m}').format(t=w['tenure'], m=w['monthly']) + (T(' · mismo método de pago') if w['payment'] else '')
    risk = {'alto': T('Riesgo alto'), 'medio': T('Riesgo medio'), 'bajo': T('Riesgo bajo')}[lv]
    return card(html.Div([html.Div([signal(r), html.Strong(risk, style={'color': col}), html.Small(T('{n} clientes parecidos').format(n=f'{len(s):,}'))], className='sim-sig'),
                          graph(gauge(r, T('Abandono observado en clientes similares'), col), 260, False, GAUGE_M)], className='sim-top'),
                html.P([T('Criterio de similitud: mismo contrato e internet, '), window, '.'], className='insight'),
                html.Div([html.Span(T('{n} se fueron').format(n=int(s.target.sum())), className='pill coral'), html.Span(T('{n} se quedaron').format(n=int((1 - s.target).sum())), className='pill cyan')], className='pc-pills'))


@app.callback(Output('real-id', 'value'), Input('real-random', 'n_clicks'), prevent_initial_call=True)
def real_random(n): return int(np.random.default_rng(n).choice(a.TEST_IDS))


@app.callback(Output('real-output', 'children'), Input('real-id', 'value'), State('lang', 'data'))
def real_output(row, lang='es'):
    set_lang(lang)
    c = a.RAW.loc[row]; sc = a.test_scores().loc[row]
    facts = [(T('Contrato'), cat(c.Contract)), ('Internet', cat(c.InternetService)), (T('Pago'), cat(c.PaymentMethod)), (T('Antigüedad'), f'{c.tenure} ' + T('meses')),
             (T('Cargo mensual'), f'${c.MonthlyCharges:.2f}'), (T('Soporte técnico'), cat(c.TechSupport)), (T('Seguridad en línea'), cat(c.OnlineSecurity)), (T('Adulto mayor'), cat(int(c.SeniorCitizen)))]
    gauges = [graph(gauge(sc[e], T(a.FINALIST_NAMES[e]), level(sc[e])[1]), 220, False, GAUGE_M) for e in a.FINALISTS]
    verdict = html.Div([icon('x' if c.target else 'check', 'sm'), T('En la realidad este cliente '), html.B(T('abandonó') if c.target else T('permaneció'))], className='truth ' + ('coral' if c.target else 'cyan'))
    return html.Div([html.Div([html.Div([html.Div([html.Small(T('Cliente')), html.Strong(c.customerID)], className='pc-row'),
                                         html.Div([html.Div([html.Small(k), html.Strong(v)]) for k, v in facts], className='facts'), verdict], className='mini-phone'),
                               html.Div(gauges, className='gauges')], className='real'),
                     html.P(T('Probabilidades nativas guardadas de los modelos finalistas, sin recalibrar. El umbral por defecto es 0.5.'), className='insight')])


@app.callback(Output('threshold-output', 'children'), Input('threshold-model', 'value'), Input('threshold', 'value'), State('lang', 'data'))
def threshold_output(experiment, threshold, lang='es'):
    set_lang(lang)
    p = a.predictions(experiment); cm, m = a.threshold_metrics(p, threshold)
    tn, fp, fn, tp = cm.ravel(); total = cm.sum()
    quota = {k: round(v / total * 100) for k, v in {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}.items()}
    quota['tn'] += 100 - sum(quota.values())
    legend = [('tp', T('Abandonos detectados'), tp), ('fn', T('Abandonos que se escapan'), fn), ('fp', T('Falsas alertas'), fp), ('tn', T('Clientes estables bien leídos'), tn)]
    names = {k: t for k, t, _ in legend}
    waffle = [html.Span(className='w ' + k, title=names[k]) for k in ['tp', 'fn', 'fp', 'tn'] for _ in range(max(0, quota[k]))]
    fpr, tpr, _ = roc_curve(p.y_true, p.score)
    roc = go.Figure(go.Scatter(x=fpr, y=tpr, line_color=BASE, name='ROC', fill='tozeroy', fillcolor='rgba(46,134,171,.10)'))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line={'dash': 'dash', 'color': GREY}, name=T('Azar')))
    roc.add_trace(go.Scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode='markers', marker={'size': 14, 'color': HOT, 'line': {'color': 'white', 'width': 2}}, name=T('Umbral actual')))
    roc.update_xaxes(title=T('Tasa de falsas alertas')); roc.update_yaxes(title=T('Tasa de detección'))
    cards = [kpi(k.upper() if k != 'Alertas' else T('ALERTAS'), round(v * 100, 1) if k != 'Alertas' else v, T('test histórico'), '', 1 if k != 'Alertas' else 0, ' %' if k != 'Alertas' else '')
             for k, v in m.items()]
    return html.Div([kpis(*cards),
                     grid(card(html.H3(T('100 clientes del test'), className='chart-title'), html.Div(waffle, className='waffle'),
                               html.Div([html.Div([html.Span(className='w ' + k), f'{t}: ', html.B(f'{v:,}')], className='leg') for k, t, v in legend], className='legend')),
                          chart(T('Curva ROC y punto de operación'), roc, T('AUC = {a}. El punto resaltado se mueve con el umbral.').format(a=f'{roc_auc_score(p.y_true, p.score):.3f}'), 360))])


@app.callback(Output('explorer-table', 'children'), Input('explorer-cols', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'), State('lang', 'data'))
def explorer_table(cols, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    return table(d[cols or ['customerID']], id='data-table', page_size=12) if not d.empty else empty()


@app.callback(Output('download', 'data'), Input('export', 'n_clicks'), State('explorer-cols', 'value'), State('split', 'value'), State('contract', 'value'), State('internet', 'value'), State('tenure', 'value'), prevent_initial_call=True)
def export(n, cols, split, contract, internet, tenure):
    # Solo un clic real descarga: al volver a pintar la sección el botón nace con n_clicks = 0.
    if not n: return no_update
    return dcc.send_data_frame(a.cohort(split, contract, internet, tenure)[cols or ['customerID']].to_csv, 'telco_seleccion.csv', index=False)


chatbot.register(app)

# Tema e idioma en el navegador: se guardan en localStorage (dcc.Store) y se aplican al <html>.
app.clientside_callback("function(n, t){ if(!n) return window.dash_clientside.no_update; return t === 'light' ? 'dark' : 'light'; }",
                        Output('theme', 'data'), Input('theme-btn', 'n_clicks'), State('theme', 'data'), prevent_initial_call=True)
app.clientside_callback("function(t){ document.documentElement.dataset.theme = t || 'dark'; return t || 'dark'; }", Output('theme-btn', 'value'), Input('theme', 'data'))
app.clientside_callback("function(h, l){window.scrollTo({top:0,behavior:'smooth'});document.documentElement.lang=l||'es';var k=(h||'#inicio').slice(1);"
                        "var T=" + json.dumps({'es': TITLES, 'en': {k: EN.get(v, v) for k, v in TITLES.items()}}) + "[l==='en'?'en':'es'];"
                        "document.title=(T[k]||T.inicio)+(l==='en'?' · Lost signal':' · Señal perdida');return '';}",
                        Output('footer', 'title'), Input('url', 'hash'), Input('lang', 'data'))

if __name__ == '__main__': app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=False)
