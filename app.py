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

BOOK = 'https://nataly-cardenas.github.io/Telco_Churn_JBook/'
REPO = 'https://github.com/Nataly-Cardenas/Telco_Churn_JBook'
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


def bar_labels(f, values, cats, at, fmt='{:.3f}', horizontal=True):
    """Valores escritos al inicio visible de cada barra, lejos de los bigotes de error (el eje no empieza en cero)."""
    kw = dict(x=[at] * len(values), y=list(cats), textposition='middle right') if horizontal else dict(x=list(cats), y=[at] * len(values), textposition='top center')
    f.add_trace(go.Scatter(mode='text', text=[fmt.format(v) for v in values], textfont={'color': '#ffffff', 'size': 13}, hoverinfo='skip', showlegend=False, **kw))
    return f


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
    # La altura la pone el contenedor (graph): así cada gráfica cabe completa en la pantalla al proyectar.
    below = f.layout.legend.y is not None and f.layout.legend.y < 0   # leyenda pedida debajo de la gráfica
    if margin is None and f.layout.margin.b is not None: margin = f.layout.margin.to_plotly_json()   # margen propio de la figura
    f.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter, sans-serif', 'color': '#8792a8', 'size': 13.5}, colorway=[BASE, HOT, GREY],
                    margin=margin or {'l': 20, 'r': 20, 't': 36, 'b': 30}, height=None, autosize=True, showlegend=legend,
                    legend={'orientation': 'h', 'y': f.layout.legend.y if below else 1.14, 'yanchor': 'top' if below else 'auto', 'title_text': '', 'bgcolor': 'rgba(0,0,0,0)'},
                    transition={'duration': 600, 'easing': 'cubic-in-out'},
                    hoverlabel={'bgcolor': '#0b1426', 'bordercolor': BASE, 'font': {'family': 'Inter', 'color': '#eaf2ff', 'size': 13}})
    f.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID); f.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    # Ejes que Plotly crea al dibujar (matriz de dispersión) heredan la rejilla de la plantilla: se ajusta también ahí.
    for ax in (f.layout.template.layout.xaxis, f.layout.template.layout.yaxis): ax.update(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return f


def graph(f, height=360, legend=True, margin=None):
    fit = f'min({height}px, calc(100vh - 230px))'   # nunca más alta que la pantalla disponible bajo la barra superior
    return dcc.Graph(figure=style(f, height, legend, margin), style={'height': fit, 'minHeight': f'{min(height, 240)}px', 'width': '100%'},
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
    univariado = [grid(chart(T('Variable objetivo'), target_figure(d), T('Clientes que permanecen y que abandonan: el desbalance obliga a usar métricas de la clase positiva.'), 340, False),
                       card(html.H3(T('Estadísticas descriptivas'), className='chart-title'), table(tt(eda.numeric_summary(d))))),
                  subhead(T('Variables numéricas')),
                  card(field(T('Variable'), dropdown('uni-num', a.NUMERIC, 'tenure', num_opts)), html.Div(id='uni-num-output'), cls='controls-card'),
                  subhead(T('Variables categóricas')),
                  card(field(T('Variable'), dropdown('uni-cat', ['__all__'] + a.CATEGORICAL, '__all__', {'__all__': T('Todas las variables'), **cat_opts})), html.Div(id='uni-cat-output'), cls='controls-card'),
                  subhead(T('Outliers y sesgos')),
                  card(table(tt(eda.outliers(d))), html.P(T('Con el criterio IQR ninguna variable numérica presenta valores atípicos; TotalCharges tiene asimetría positiva. No se eliminan observaciones.'), className='insight'))]
    bivariado = [subhead(T('Variable objetivo vs variables numéricas')),
                 card(field(T('Variable'), dropdown('bi-num', a.NUMERIC, 'tenure', num_opts)), html.Div(id='bi-num-output'), cls='controls-card'),
                 html.Div(id='bi-pb-output'),
                 subhead(T('Variable objetivo vs variables categóricas')),
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
                     tabs('eda-tabs', [(T('1 · ETL y partición'), etl), (T('2 · Univariado'), univariado), (T('3 · Bivariado'), bivariado), (T('4 · Multivariado'), multivariado),
                                       (T('5 · Valores faltantes'), faltantes), (T('6 · Resumen'), resumen), (T('7 · Explorador de datos'), explorador)])])


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
STEPS = [('target', N('Objetivo'), N('Churn Yes/No → 1/0. customerID solo para control.')), ('shield', N('TotalCharges'), N('Texto a número; los 8 vacíos de train (3 de test) tienen tenure 0 → 0.')),
         ('layers', N('Ingeniería'), N('“Sin internet / sin teléfono” → “No” y AvgChargePerMonth = TotalCharges / tenure.')), ('table', N('Codificar y escalar'), N('4 numéricas con StandardScaler y 16 categóricas con One-Hot: 24 columnas.')),
         ('scale', N('Balancear'), N('Ninguno, pesos, SMOTE o ADASYN; solo dentro del entrenamiento.')), ('network', N('Modelo'), N('Se ajusta en entrenamiento y se evalúa en el fold reservado.'))]
COEFS = [('cat__Contract_Two year', -1.422598), ('num__tenure', -1.153496), ('cat__InternetService_No', -1.079819), ('cat__InternetService_Fiber optic', 1.036427), ('cat__Contract_One year', -0.725538),
         ('num__TotalCharges', 0.486039), ('cat__PaymentMethod_Electronic check', 0.398853), ('cat__OnlineSecurity_Yes', -0.371758), ('cat__StreamingMovies_Yes', 0.348344), ('cat__PaperlessBilling_Yes', 0.334539),
         ('cat__StreamingTV_Yes', 0.323924), ('cat__PhoneService_Yes', -0.31652), ('cat__TechSupport_Yes', -0.304333), ('cat__MultipleLines_Yes', 0.28687), ('num__MonthlyCharges', -0.272618),
         ('cat__Dependents_Yes', -0.231701), ('cat__SeniorCitizen_1', 0.153587), ('cat__OnlineBackup_Yes', -0.110944), ('cat__PaymentMethod_Mailed check', 0.0482), ('num__AvgChargePerMonth', -0.046767),
         ('cat__gender_Male', 0.029063), ('cat__Partner_Yes', 0.019222), ('cat__PaymentMethod_Credit card (automatic)', -0.00774), ('cat__DeviceProtection_Yes', 0.004777)]


def preprocesamiento(d):
    imput = pd.DataFrame({'m': [T('Imputar con 0'), T('Mediana + indicador')], 'auc': [.8459, .8460], 'sd': [.0124, .0125]})
    fi = go.Figure(go.Bar(x=imput.m, y=imput.auc, error_y={'type': 'data', 'array': imput.sd, 'color': GREY}, marker_color=[HOT, BASE]))
    bar_labels(fi, imput.auc, imput.m, .802, '{:.4f}', horizontal=False)
    fi.update_yaxes(range=[.8, .87], title='AUC-ROC CV')
    pasos = [card(html.H3(T('El pipeline paso a paso'), className='chart-title'), html.Div([html.Div(className='flow-line'), *[html.Div([html.Span(icon(i), className='st-icon'), html.H4(T(t)), html.P(T(s))], className='station',
                  style={'animationDelay': f'{k * .15}s'}) for k, (i, t, s) in enumerate(STEPS)]], className='pipeline')),
             grid(chart(T('¿Cero o mediana para TotalCharges?'), fi, T('Diferencia pareada por fold: +0.0001 ± 0.0001. Imputar con 0 es coherente con clientes sin facturación y no pierde desempeño.'), 320, False),
                  card(html.H3(T('Ingeniería de características'), className='chart-title'),
                       kpis(kpi(T('CORRELACIÓN CON MONTHLYCHARGES'), .9963, 'AvgChargePerMonth', '', 4), kpi(T('COLUMNAS TRAS CODIFICAR'), 24, T('4 numéricas + 20 indicadores')), cls='k2'),
                       html.P(T('El cargo promedio mensual comparte casi toda la información con el cargo mensual; su aporte se evalúa en la ablación.'), className='insight')))]
    cv = pd.DataFrame({'m': [T('Regresión logística'), T('Etiquetas permutadas'), T('Dummy estratificado'), T('Dummy mayoritaria')], 'auc': [.8454, .4868, .5065, .5], 'sd': [.0123, .0220, .0171, 0]})
    fc = go.Figure(go.Bar(x=cv.auc, y=cv.m, orientation='h', error_x={'type': 'data', 'array': cv.sd, 'color': GREY}, marker_color=highlight(cv.auc), text=cv.auc.map('{:.4f}'.format), textposition='inside', insidetextanchor='start'))
    fc.update_xaxes(range=[0, 1], title='AUC-ROC CV (5 folds)'); fc.add_vline(x=.5, line_dash='dot', line_color=GREY)
    validacion = [kpis(kpi('AUC-ROC CV', .8454, '± 0.0123', '', 4), kpi('PR-AUC CV', .6581, '± 0.0191', '', 4), kpi('RECALL CV', .8020, '± 0.0341', '', 4), kpi(T('CONTROL PERMUTADO'), .4868, T('AUC del azar'), '', 4)),
                  chart(T('Validación cruzada en entrenamiento'), fc, T('El modelo supera con claridad a los controles: con etiquetas permutadas o un clasificador trivial el AUC ronda 0.5.'), 340, False)]
    abl = pd.DataFrame({'c': [T('Sin gender'), T('Sin PhoneService'), T('Sin PhoneService ni MultipleLines'), T('Sin gender ni PhoneService'), T('Sin AvgChargePerMonth')],
                        'd': [.0001, .0002, -.0007, .0004, .0005], 'sd': [.0005, .0003, .0018, .0004, .0003], 'dpr': [-.0003, -.0003, -.0106, -.0000, .0019]})
    fa = go.Figure(go.Bar(x=abl.d, y=abl.c, orientation='h', error_x={'type': 'data', 'array': abl.sd, 'color': GREY}, marker_color=BASE))
    fa.add_vline(x=0, line_color=GREY); fa.update_xaxes(title=T('Δ AUC-ROC frente al modelo completo'), tickformat='.4f')
    ablacion = [chart(T('Ablación por validación cruzada'), fa, T('Retirar gender, PhoneService o AvgChargePerMonth cambia el AUC menos que la variación entre folds; las variables se conservan.'), 340, False),
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
            grid(chart(T('Matriz de confusión'), fcm, T('Detecta 293 de 374 abandonos, pero 286 de sus 579 alertas son falsas: hay margen de mejora.'), 360, False),
                 chart(T('Curva ROC'), froc, T('Pulsa play para trazar la curva. AUC = 0.8419 en el test.'), 360)),
            chart(T('Coeficientes de la regresión logística'), fco, T('Contrato a dos años, antigüedad y no tener internet reducen el riesgo; la fibra óptica lo aumenta. gender y PhoneService (resaltadas) pesan poco.'), 640, False)]
    return html.Div([page_head('preprocesamiento', T('Cómo se transforman los datos antes de modelar, sin espiar el test, y cuánto logra el modelo base.')),
                     tabs('prep-tabs', [(T('1 · Pasos'), pasos), (T('2 · Validación cruzada'), validacion), (T('3 · Ablación'), ablacion), (T('4 · Evaluación en test'), test)])])


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
    resultados.append(robustez())
    calibracion = [calibration_summary()]
    optimizadores = [card(field(T('Familia'), dropdown('opt-family', list(a.MODELS), 'random_forest', fam)), cls='controls-card'), html.Div(id='opt-output'), optimizer_effects()]
    return html.Div([page_head('modelado', T('Siete familias × cuatro balanceos × cuatro optimizadores = 112 configuraciones evaluadas con validación cruzada anidada.')),
                     html.Div([podium(), kpis(kpi(T('CONFIGURACIONES'), 112, '7 × 4 × 4'), kpi(T('UNIDADES EXTERNAS'), 1680, T('112 × 3 semillas × 5 folds')),
                                              kpi('FOLDS', '5 × 3', T('externos × internos')), kpi(T('PRESUPUESTO'), 24, T('evaluaciones por búsqueda')), cls='k2')], className='hero'),
                     tabs('model-tabs', [(T('1 · Resultados por modelo'), resultados), (T('2 · Calibración'), calibracion), (T('3 · Optimizadores'), optimizadores),
                                         (T('4 · Multifidelidad'), multifidelidad()), (T('5 · Eficiencia computacional'), eficiencia()),
                                         (T('6 · Comparación estadística'), estadistica() + effect_sizes()), (T('7 · Interpretabilidad'), interpretabilidad())])])


def ranking_families():
    s = a.SUMMARY.set_index('experiment_id').loc[list(BEST_BY_FAMILY.values())].reset_index().sort_values('f1_positive_mean')
    s['n'] = s.experiment_id.map(config_name)
    f = go.Figure(go.Bar(x=s.f1_positive_mean, y=s.n, orientation='h', error_x={'type': 'data', 'array': s.f1_positive_std, 'color': GREY},
                         marker_color=highlight(s.f1_positive_mean, len(s) - 1)))
    bar_labels(f, s.f1_positive_mean, s.n, .551)
    f.update_xaxes(range=[.55, .68], title=T('F1 positivo · media de 5 folds externos'))
    return chart(T('Mejor configuración de cada familia'), f, T('Barras: media de los folds externos; bigotes: desviación estándar. Random Forest y XGBoost empatan arriba (0.638).'), 380, False)


# ================================================================ 4. MODELADO AVANZADO · resultados complementarios del libro
COMP = a.DATA / 'complementarios'
SOFT = 'rgba(46,134,171,.45)'   # el mismo azul, más suave: la versión de referencia (sin calibrar, estándar)


@lru_cache(maxsize=None)
def comp(name): return pd.read_csv(COMP / f'{name}.csv')


def robustez():
    r = comp('robustez_semillas').copy(); r['n'] = r.model.map(model_name); r = r.sort_values('f1_positive_mean')
    f = go.Figure(go.Scatter(x=r.f1_positive_mean, y=r.n, mode='markers', name=T('Realización principal · IC 95 %'), marker={'size': 13, 'color': highlight(r.f1_positive_mean)},
                             error_x={'type': 'data', 'symmetric': False, 'array': r.high - r.f1_positive_mean, 'arrayminus': r.f1_positive_mean - r.low, 'color': GREY}))
    for k, sym in [(1, 'diamond'), (2, 'square')]:
        f.add_trace(go.Scatter(x=r[f'F1 semilla {k}'], y=r.n, mode='markers', name=T('Semilla {k}').format(k=k), marker={'size': 9, 'color': SOFT, 'symbol': sym}))
    f.update_xaxes(title=T('F1 positivo · media de 5 folds externos'))
    t = r[['n', 'f1_positive_mean', 'f1_positive_std', 'F1 semilla 1', 'F1 semilla 2', 'low', 'high']].rename(
        columns={'n': 'Familia', 'f1_positive_mean': 'F1 realización principal', 'f1_positive_std': 'DE entre folds', 'low': 'IC 95 % inferior', 'high': 'IC 95 % superior'})
    return html.Div([subhead(T('Robustez entre semillas e intervalos de confianza')),
                     grid(chart(T('F1 de cada familia en las tres realizaciones'), f, T('Las semillas cambian el optimizador, el modelo y el remuestreo con los mismos folds: el F1 apenas se mueve. Los intervalos son condicionales a modelos fijos.'), 380),
                          card(html.H3(T('Media por realización e IC condicional'), className='chart-title'), table(tt(t[::-1].round(4))),
                               html.P(T('Los intervalos usan 2.000 remuestreos estratificados por fold; no incluyen reentrenamiento ni la selección entre 112 configuraciones.'), className='insight')))])


def calibration_summary():
    rep = comp('calibracion_representantes').copy(); rep['n'] = rep.model.map(model_name)

    def bars(kind):
        f = go.Figure([go.Bar(x=rep.n, y=rep[f'{kind} nativo'], name=T('Nativa'), marker_color=SOFT),
                       go.Bar(x=rep.n, y=rep[f'{kind} sigmoid'], name=T('Sigmoide (Platt)'), marker_color=BASE),
                       go.Bar(x=rep.n, y=rep[f'{kind} isotónica'], name=T('Isotónica'), marker_color=HOT)])
        f.update_layout(barmode='group'); f.update_yaxes(title=kind)
        return f
    u = a.UNITS[a.UNITS.replicate == 0].groupby('experiment_id')[['cal_brier', 'cal_ece']].mean()
    m = u.join(comp('isotonica_unidades').groupby('experiment_id')[['iso_brier', 'iso_ece']].mean()).reset_index()
    m['n'] = m.experiment_id.map(config_name)

    def scatter(x, y, title):
        better = m[y] < m[x]; lo = min(m[x].min(), m[y].min()) * .97; hi = max(m[x].max(), m[y].max()) * 1.03
        f = go.Figure(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', line={'dash': 'dash', 'color': GREY}, showlegend=False, hoverinfo='skip'))
        for mask, name, color in [(better, T('Isotónica mejor'), HOT), (~better, T('Sigmoide mejor'), BASE)]:
            f.add_trace(go.Scatter(x=m[x][mask], y=m[y][mask], mode='markers', name=f'{name} ({int(mask.sum())})', marker={'color': color, 'size': 8, 'opacity': .8},
                                   text=m.n[mask], hovertemplate='%{text}<br>' + T('sigmoide') + ' %{x:.4f} · ' + T('isotónica') + ' %{y:.4f}<extra></extra>'))
        f.update_xaxes(title=f"{title} · {T('sigmoide (Platt)')}", range=[lo, hi]); f.update_yaxes(title=f"{title} · {T('isotónica')}", range=[lo, hi])
        return f
    fam = {k: model_name(k) for k in a.MODELS}
    return html.Div([
        kpis(kpi(T('ISOTÓNICA MEJORA A LA NATIVA'), 84, T('de 96 configuraciones, en Brier y ECE')), kpi(T('PLATT MEJORA A LA NATIVA'), 80, T('de 96 configuraciones, en Brier y ECE')),
             kpi(T('ISOTÓNICA CON MENOR ECE'), 74, T('de 112 frente a Platt')), kpi(T('ISOTÓNICA CON MENOR BRIER'), 67, T('de 112 frente a Platt'))),
        note(T('¿Por qué recalibrar?'), T('Balancear con SMOTE o con pesos le enseña al modelo un mundo donde la mitad de los clientes abandona, así que sus probabilidades quedan infladas. '
                                         'Platt ajusta una curva en S con dos parámetros; la isotónica ajusta una escalera no decreciente (PAV) sin suponer ninguna forma. '
                                         'Ambas usan las mismas tres particiones dentro del entrenamiento externo: el fold externo nunca entrena al calibrador.')),
        card(field(T('Familia (mejor configuración)'), dropdown('cal-family', list(BEST_BY_FAMILY), 'random_forest', fam)), html.Div(id='cal-map-output'), cls='controls-card'),
        grid(chart(T('Brier por familia · representantes'), bars('Brier'), T('Menor es mejor. Las dos recalibraciones recortan el Brier entre 9 % y 34 %; entre ellas la diferencia casi siempre está en la cuarta cifra decimal.'), 360),
             chart(T('ECE por familia · representantes'), bars('ECE'), T('Las dos ramas bajan el ECE de 0.10–0.21 a 0.02–0.04. SVM no tiene probabilidad nativa: solo existe recalibrada.'), 360)),
        grid(chart(T('Brier de las 112 configuraciones'), scatter('cal_brier', 'iso_brier', 'Brier'), T('Cada punto es una configuración; bajo la diagonal (resaltados) la isotónica gana. Casi todos quedan pegados a la diagonal.'), 380),
             chart(T('ECE de las 112 configuraciones'), scatter('cal_ece', 'iso_ece', 'ECE'), T('La isotónica tiene menor ECE en 74 de 112 configuraciones.'), 380)),
        note(T('Síntesis de la calibración'), T('Conviene la isotónica cuando la distorsión no tiene forma de S (GaussianNB, árboles) y Platt cuando el score ya es un margen o una fracción de vecinos (SVM, KNN). '
                                               'Ojo con el F1: calibrar acerca las probabilidades a la prevalencia real y, con el umbral fijo en 0.5, ambas ramas detectan menos abandonos. '
                                               'La calibración sirve para leer las probabilidades como riesgos; no se usó para elegir el ranking principal.'))])


def optimizer_effects():
    e = comp('efecto_optimizadores'); opts = list(dict.fromkeys(e.optimizer)); names = [T(o) for o in opts]
    f = go.Figure()
    for ev, color in [(8, HOT), (24, BASE)]:
        s = e[e.evaluaciones == ev].set_index('optimizer').loc[opts]
        f.add_trace(go.Bar(x=names, y=s.d_z_pareado, name=T('{n} evaluaciones').format(n=ev), marker_color=color, text=s.d_z_pareado.map('{:+.2f}'.format), textposition='outside'))
    f.update_layout(barmode='group'); f.add_hline(y=0, line_color=GREY); f.update_yaxes(title=T('d_z pareado frente a Grid'), range=[-.35, .6])
    t = e.rename(columns={'evaluaciones': 'Evaluaciones', 'optimizer': 'Optimizador', 'comparaciones': 'Comparaciones', 'delta_f1_interno': 'ΔF1 interno',
                          'cliff_no_pareado': 'δ de Cliff (sin emparejar)', 'd_z_pareado': 'd_z pareado', 'rank_biserial_pareado': 'Biserial de rangos', 'magnitud_cliff': 'Magnitud'})
    c = comp('coste_busqueda').copy(); c['model'] = c.model.map(model_name); c['optimizer'] = c.optimizer.map(optimizer_name)
    c = c.rename(columns={'model': 'Familia', 'optimizer': 'Optimizador', 'search_mean': 'Búsqueda (s)', 'search_std': 'DE búsqueda', 'refit_mean': 'Reajuste (s)',
                          'inference_mean': 'Inferencia (s)', 'rss_mean_mib': 'Memoria (MiB)', 'evaluations': 'Evaluaciones', 'proposals': 'Propuestas', 'seconds_per_evaluation': 'Segundos por evaluación'})
    return html.Div([subhead(T('Tamaño del efecto frente a Grid')),
                     grid(chart(T('Ventaja pareada de cada optimizador sobre Grid'), f, T('Con 8 evaluaciones (resaltado) hay una ventaja pequeña pero consistente (d_z ≈ 0.41–0.47); con 24 desaparece y queda del orden de 10⁻⁴ en F1.'), 340),
                          card(html.H3(T('420 búsquedas emparejadas por optimizador y presupuesto'), className='chart-title'), table(tt(t.round(4))),
                               html.P(T('Sin emparejar, el δ de Cliff es despreciable (|δ| ≤ 0.07): la variación entre familias y folds es mucho mayor que la del optimizador.'), className='insight'))),
                     details([icon('table'), T('Presupuesto y coste empírico por familia y optimizador')], table(tt(c.round(4)), page_size=28))])


MF_NAMES = {'successive_halving': N('Successive Halving'), 'random_completo': N('Random completo')}


def multifidelidad():
    r = comp('multifidelidad_resumen'); pf = comp('multifidelidad_por_fold')
    funnel = go.Figure(go.Funnel(y=[T('Ronda {k}').format(k=k) for k in range(1, 5)], x=[27, 9, 3, 1], text=['r/27', 'r/9', 'r/3', 'r'], textinfo='value+text',
                                 marker={'color': [BASE, BASE, BASE, HOT]}, connector={'line': {'color': GREY, 'width': 1}}))
    cards = []
    for m in ['random_forest', 'svm', 'xgboost']:
        s = r[r.model == m].set_index('method'); speed = s.loc['random_completo', 'segundos'] / s.loc['successive_halving', 'segundos']
        d = s.loc['successive_halving', 'f1_externo'] - s.loc['random_completo', 'f1_externo']
        cards.append(kpi(model_name(m).upper(), round(speed, 1), T('ΔF1 externo {d} · Spearman entre rondas {s}').format(d=f'{d:+.3f}', s=f"{s.loc['successive_halving', 'spearman_rondas']:.2f}"),
                         'hot' if m == 'svm' else '', 1, '×'))
    f = go.Figure()
    for m, sym in [('random_forest', 'circle'), ('svm', 'diamond'), ('xgboost', 'square')]:
        for meth, color in [('random_completo', BASE), ('successive_halving', HOT)]:
            s = pf[(pf.model == m) & (pf.method == meth)]
            f.add_trace(go.Scatter(x=s.search_seconds, y=s.outer_f1, mode='markers', name=T(MF_NAMES[meth]), legendgroup=meth, showlegend=m == 'random_forest',
                                   marker={'color': color, 'symbol': sym, 'size': 11, 'opacity': .85},
                                   hovertemplate=f'{model_name(m)} · {T(MF_NAMES[meth])}<br>' + T('Fold') + ' %{text}<br>%{x:.0f} s · F1 %{y:.3f}<extra></extra>', text=s.outer_fold))
    f.update_xaxes(type='log', title=T('Tiempo de búsqueda por fold externo (s, escala log)')); f.update_yaxes(title=T('F1 en el fold externo'))
    t = r.copy(); t['model'] = t.model.map(model_name); t['method'] = t.method.map(lambda x: T(MF_NAMES[x]))
    t = t.rename(columns={'model': 'Familia', 'method': 'Método', 'segundos': 'Segundos', 'segundos_ds': 'DE segundos', 'ajustes': 'Ajustes', 'ajustes_equivalentes': 'Ajustes equivalentes',
                          'f1_interno': 'F1 interno', 'f1_externo': 'F1 externo', 'f1_externo_ds': 'DE F1 externo', 'auc_externo': 'AUC externo', 'spearman_rondas': 'Spearman entre rondas',
                          'f1_random_seccion_1': 'F1 Random (sección 1)'})
    return [kpis(*cards, kpi(T('CÓMPUTO TEÓRICO'), 6, T('12 ajustes completos frente a 72'), '', 0, '×'), cls='k4'),
            grid(chart(T('Successive Halving: 27 candidatos, 4 rondas'), funnel, T('En cada ronda pasa el tercio con mejor F1 interno y el recurso se triplica: árboles en Random Forest y XGBoost (12 a 324), filas en SVM.'), 340, False),
                 chart(T('Calidad frente a costo: cada punto es un fold externo'), f, T('Círculos: Random Forest; rombos: SVM; cuadrados: XGBoost. En Random Forest y SVM los puntos de SH (resaltados) quedan a la izquierda sin bajar: igual F1 con una fracción del tiempo. En XGBoost SH eligió peor en cuatro de cinco folds.'), 340)),
            note(T('¿Cuándo conviene la multifidelidad?'), T('Cuando el recurso recortado domina el costo y no altera el orden de los candidatos: filas en SVM (su costo crece como n²) y árboles en Random Forest. '
                                                             'En XGBoost la tasa de aprendizaje interactúa con el número de árboles: con 12 árboles un buen candidato parece malo, y la fidelidad baja engaña.')),
            details([icon('table'), T('Resumen por familia y método (5 folds externos)')], table(tt(t.round(4))))]


def profile_chart(name, core):
    p = comp(name).head(8).copy()
    p['f'] = p['filename:lineno(function)'].str.replace(r'^.*/', '', regex=True).str.replace(r':\d+\(', ' · ', regex=True).str.rstrip(')')
    p = p.drop_duplicates('f')[::-1]
    f = go.Figure(go.Bar(x=p.cumtime, y=p.f, orientation='h', marker_color=[HOT if core in x else BASE for x in p.f], text=p.cumtime.map('{:.3f} s'.format), textposition='outside', cliponaxis=False))
    f.update_xaxes(title=T('Tiempo acumulado (s)'), range=[0, p.cumtime.max() * 1.25])
    return f


VARIANT_FAMILIES = ['svm', 'xgboost', 'random_forest', 'knn', 'logistic', 'naive_bayes']


def eficiencia():
    e = comp('coste_empirico').copy(); e['n'] = e.model.map(model_name)
    fe = go.Figure([go.Bar(x=e.n, y=e.refit_seconds, name=T('Reajuste'), marker_color=BASE), go.Bar(x=e.n, y=e.predict_seconds, name=T('Predicción del fold externo'), marker_color=HOT)])
    fe.update_layout(barmode='group'); fe.update_yaxes(title=T('Segundos (media de las unidades)'))
    th = comp('complejidad_teorica').copy(); th['model'] = th.model.map(model_name)
    th = th.rename(columns={'model': 'Familia', 'training_O': 'Entrenamiento', 'prediction_O': 'Predicción', 'memory_O': 'Memoria', 'assumptions': 'Supuestos'})
    ex = comp('escalamiento_exponentes'); ex = ex.sort_values('exponente_n')
    fx = go.Figure([go.Bar(x=ex.exponente_n, y=ex.model, orientation='h', name=T('Exponente en n'), marker_color=highlight(ex.exponente_n), text=ex.exponente_n.map('{:.2f}'.format), textposition='outside', cliponaxis=False)])
    fx.add_vline(x=1, line_dash='dot', line_color=GREY, annotation_text=T('lineal'), annotation_position='top'); fx.add_vline(x=2, line_dash='dot', line_color=GREY, annotation_text=T('cuadrático'), annotation_position='top')
    fx.update_xaxes(title=T('Exponente empírico (pendiente log-log)'), range=[-1, 2.6])
    rf = comp('estandar_vs_optimizado'); rf = rf[rf.model == 'random_forest']
    par = comp('paralelismo_resumen')
    fp = go.Figure([go.Scatter(x=[1, 2, 4, 8], y=[1, 2, 4, 8], mode='lines', name=T('Ideal lineal'), line={'dash': 'dash', 'color': GREY}),
                    go.Scatter(x=[1, 2, 4, 8], y=rf['aceleración_entrenamiento'], mode='lines+markers', name=T('Random Forest: entrenamiento (hilos)'), line={'color': BASE, 'width': 3}),
                    go.Scatter(x=[1, 2, 4, 8], y=rf['aceleración_inferencia'], mode='lines+markers', name=T('Random Forest: inferencia (hilos)'), line={'color': SOFT, 'width': 2, 'dash': 'dot'}),
                    go.Scatter(x=par[par.backend == 'loky'].n_jobs, y=par[par.backend == 'loky'].speedup, mode='lines+markers', name=T('Validación cruzada RF–SMOTE (loky)'), line={'color': HOT, 'width': 3}),
                    go.Scatter(x=[1, *par[par.backend == 'threading'].n_jobs], y=[1, *par[par.backend == 'threading'].speedup], mode='lines+markers', name=T('Validación cruzada RF–SMOTE (threading)'), line={'color': HOT, 'width': 2, 'dash': 'dash'})])
    fp.update_xaxes(title=T('Trabajadores'), tickvals=[1, 2, 4, 8]); fp.update_yaxes(title=T('Aceleración frente a 1 trabajador'))
    fp.update_layout(legend={'y': -.24}, margin={'l': 20, 'r': 20, 't': 20, 'b': 120})
    cmp_ = comp('estandar_vs_optimizado').copy(); cmp_['model'] = cmp_.model.map(model_name)
    cmp_ = cmp_.rename(columns={'model': 'Familia', 'variant': 'Variante', 'complejidad': 'Complejidad', 'entrenamiento_s': 'Entrenamiento (s)', 'inferencia_s': 'Inferencia (s)',
                                'memoria_entrenamiento_mib': 'Memoria (MiB)', 'aceleración_entrenamiento': 'Aceleración entrenamiento', 'aceleración_inferencia': 'Aceleración inferencia', 'delta_f1': 'ΔF1'})
    bench = comp('benchmark_trabajadores').copy(); bench['model'] = bench.model.map(model_name)
    bench = bench.rename(columns={'model': 'Familia', 'requested_jobs': 'Trabajadores pedidos', 'effective_jobs': 'Trabajadores efectivos', 'total_seconds': 'Total (s)', 'total_std': 'DE',
                                  'training_seconds': 'Entrenamiento (s)', 'inference_seconds': 'Inferencia (s)', 'rss_mib': 'Memoria (MiB)', 'f1_inner': 'F1 interno'})
    variants = dropdown('eff-family', VARIANT_FAMILIES, 'svm', {k: model_name(k) for k in VARIANT_FAMILIES})
    scale_opts = list(comp('escalamiento').model.drop_duplicates())
    return [kpis(kpi('LINEARSVC', 36, T('más rápido al entrenar que SVC RBF; ΔF1 −0.002'), 'hot', 0, '×'), kpi(T('INFERENCIA LINEARSVC'), 931, T('más rápida que SVC RBF'), '', 0, '×'),
                 kpi('XGBOOST HIST', 2.8, T('más rápido que el método exacto'), '', 1, '×'), kpi(T('VALIDACIÓN CRUZADA'), 1.8, T('con 4 trabajadores y el mismo F1'), '', 1, '×')),
            subhead(T('¿Dónde se va el tiempo?'), T('cProfile sobre los dos modelos más costosos: el ajuste de SVM–SMOTE y la predicción de KNN–SMOTE.')),
            grid(chart(T('Ajuste del Pipeline SVM–SMOTE · 1.20 s'), profile_chart('perfil_svm_fit', '_dense_fit'), T('El 81 % del ajuste (0.98 s) ocurre en el solver de libsvm (resaltado); preprocesar y aplicar SMOTE suma 0.06 s.'), 330, False),
                 chart(T('Predicción del Pipeline KNN–SMOTE · 0.164 s'), profile_chart('perfil_knn_predict', 'compute'), T('El 84 % de la predicción (0.138 s) es el cálculo de distancias por pares (resaltado). El cuello de botella es el algoritmo, no el Pipeline.'), 330, False)),
            subhead(T('Implementaciones alternativas'), T('Para cada familia se entrena la versión estándar (la de las 112 configuraciones) y versiones optimizadas con los mismos hiperparámetros; mediana de 5 repeticiones por fold.')),
            card(field(T('Familia'), variants), html.Div(id='eff-variants-output'), cls='controls-card'),
            subhead(T('Escalamiento empírico frente a la complejidad teórica')),
            grid(chart(T('Exponente empírico en n (filas)'), fx, T('SVM RBF (resaltado) crece como n²: duplicar los datos cuadruplica el tiempo, como predice la teoría. Exponentes menores que 1 indican que dominan costos fijos con estos tamaños.'), 380, False),
                 card(field(T('Modelo'), dropdown('eff-scale', scale_opts, 'svm rbf', {k: k for k in scale_opts})), html.Div(id='eff-scale-output'), cls='controls-card')),
            subhead(T('Paralelización y costo medido')),
            grid(chart(T('Aceleración medida por paralelismo'), fp, T('Random Forest entrena 2× más rápido con 4 hilos y casi no mejora con 8; en inferencia los hilos empeoran el tiempo. La validación cruzada llega a 1.8× con F1 idéntico.'), 380),
                 chart(T('Reajuste y predicción por familia'), fe, T('Random Forest y SVM tienen los reajustes más costosos; SVM y KNN, la predicción más cara. Coherente con sus cotas teóricas.'), 380)),
            card(html.H3(T('Complejidad asintótica de cada familia'), className='chart-title'), table(tt(th), page_size=7),
                 html.P(T('n: tamaño tras balancear; p: columnas; T: árboles; h: profundidad; B: bins; L: hojas; m: variables por nodo; I: iteraciones; n_sv: vectores soporte. Paralelizar reduce el tiempo de pared, no estas cotas.'), className='insight')),
            card(html.H3(T('Tabla comparativa: modelo estándar frente a optimizado'), className='chart-title'), table(tt(cmp_.round(4)), page_size=21),
                 html.P(T('Tres cambios ganan claramente sin perder más de 0.002 de F1: LinearSVC (o Nyström) en lugar de SVC RBF, hist en XGBoost y paralelizar Random Forest y la validación cruzada. Quedan como recomendación para el modelo propuesto.'), className='insight')),
            note(T('Equipo y política de memoria'), T('CPU Intel Core Ultra 5 125H (14 núcleos, 18 hilos), unos 8 GB de RAM y sin GPU CUDA. Durante las 112 configuraciones la política de memoria dejó un solo trabajador efectivo, '
                                                     'por eso el paralelismo se midió después. Una caché del registro aceleró 8.08× la consulta de artefactos con resultados idénticos. XGBoost con device=cuda volvió a CPU: no se reporta aceleración por GPU.')),
            details([icon('table'), T('Benchmark de 1 y 2 trabajadores durante la corrida')], table(tt(bench.round(4)), page_size=14))]


def effect_sizes():
    b = comp('efecto_balanceo').set_index('model'); b.index = [model_name(m) for m in b.index]; cols = [T(c) for c in b.columns]
    fh = go.Figure(go.Heatmap(z=b.values, x=cols, y=list(b.index), zmin=-1, zmax=1, colorscale=DIVERGING, text=b.map('{:+.2f}'.format).values, texttemplate='%{text}',
                              colorbar={'title': 'δ', 'thickness': 14}, hovertemplate='%{y} · %{x}<br>δ = %{z:.2f}<extra></extra>'))
    fh.update_yaxes(autorange='reversed')
    fin = comp('efecto_finalistas').copy()
    fin = pd.DataFrame({T('Modelo A'): [T(a.FINALIST_NAMES[x]) for x in fin.a], T('Modelo B'): [T(a.FINALIST_NAMES[x]) for x in fin.b], T('ΔF1 medio'): fin.mean_delta_f1.round(4),
                        T('Folds con A mayor'): fin.folds_a_higher, T('δ de Cliff (5 folds)'): fin.cliff_5_folds, T('δ de Cliff (15 unidades)'): fin.cliff_15_unidades.round(3),
                        'd_z': fin.d_z_15.round(3), T('Biserial de rangos'): fin.rank_biserial_15.round(3), T('Magnitud'): fin.magnitud.map(T)})
    return [subhead(T('Tamaños del efecto'), T('Un p-valor pequeño solo dice que una diferencia es difícil de atribuir al azar, no que importe. δ de Cliff: despreciable si |δ| < 0.147, pequeño hasta 0.33, mediano hasta 0.474 y grande por encima.')),
            kpis(kpi(T('δ RF FRENTE A XGBOOST'), -0.06, T('despreciable · 15 unidades'), '', 2), kpi(T('δ ENSAMBLES FRENTE A LOGÍSTICA'), 0.10, T('despreciable · d_z 0.23–0.31'), '', 2),
                 kpi(T('PARES NEMENYI CON δ = 1'), 31, T('ΔF1 de 0.063 a 0.076 · efectos grandes'), 'hot'), kpi(T('SMOTE CON EFECTO GRANDE'), 5, T('de 7 familias (δ ≥ 0.68)'))),
            grid(chart(T('δ de Cliff del F1 frente a sin balanceo'), fh, T('Balancear tiene efecto grande en Random Forest, XGBoost, SVM y logística. GaussianNB es la excepción: ya incorpora la prevalencia y SMOTE no cambia nada.'), 380, False),
                 card(html.H3(T('Finalistas: elegir uno u otro casi da igual'), className='chart-title'), table(fin),
                      html.P(T('Los tres efectos entre finalistas son despreciables. Los 31 pares que Nemenyi separa enfrentan configuraciones balanceadas con SVM sin balanceo: son las distancias más extremas.'), className='insight')))]



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
REPRO = [('file', N('Versiones exactas'), N('requirements.txt fija cada librería usada en el entrenamiento.')),
         ('shield', N('Verificación automática'), N('verificar_reproduccion.py compara versiones, huellas SHA-256 de los datos y los artefactos de las 1,680 unidades.')),
         ('layers', N('Resultados versionados'), N('Pipelines, predicciones, checkpoints y semillas de cada unidad quedan guardados; el libro se construye sin reentrenar.')),
         ('code', N('Código del experimento'), N('Un apéndice reproduce las 32 celdas exactas que entrenaron las 112 configuraciones.'))]


def conclusiones(d):
    finds = [('layers', N('El perfil de cuenta delata el riesgo'), N('Contrato, antigüedad y servicio de internet concentran la señal, en el EDA y en SHAP.')),
             ('link', N('Empate en la cima'), N('Random Forest y XGBoost con SMOTE (F1 0.638) y la regresión logística (0.634) no se separan estadísticamente.')),
             ('scale', N('Balancear importa más que buscar'), N('SMOTE tiene efecto grande en 5 de 7 familias (δ de Cliff ≥ 0.68); cambiar de optimizador solo ayuda con presupuestos bajos.')),
             ('gauge', N('Probabilidades más honestas'), N('Platt mejoró Brier y ECE en 80 de 96 configuraciones y la isotónica en 84: isotónica para árboles y GaussianNB, Platt para SVM y KNN.')),
             ('hourglass', N('Buscar puede costar menos'), N('Successive Halving igualó el F1 de una búsqueda completa en Random Forest y SVM con 3.7× y 7.1× menos tiempo; en XGBoost no conviene.')),
             ('flame', N('El costo está en el núcleo del algoritmo'), N('LinearSVC entrena 36× y predice unas 900× más rápido que SVC RBF con 0.002 menos de F1; hist acelera 2.8× a XGBoost.'))]
    limits = [N('Un único conjunto de datos: los folds no son bloques independientes.'), N('El test ya se había usado: es evidencia suplementaria.'),
              N('Los datos son una foto: no garantizan anticipar el abandono futuro.'), N('Sin costos de intervención, el F1 no se traduce en dinero.')]
    steps = [(N('Validación temporal'), N('Evaluar en clientes futuros.')), (N('Costos explícitos'), N('Elegir el umbral con el costo de cada error.')),
             (N('Modelo propuesto'), N('Superar a las referencias en desempeño y tiempo.'))]
    return html.Div([page_head('conclusiones', T('Qué aprendimos, qué sigue siendo incierto y qué falta para llevarlo a una campaña de retención.')),
        subhead(T('Hallazgos principales')),
        html.Div([html.Div([html.Span(icon(i), className='drv-icon'), html.H3(T(t)), html.P(T(s))], className='finding') for i, t, s in finds], className='findings'),
        grid(card(eyebrow(T('LIMITACIONES')), html.Ul([html.Li(T(x)) for x in limits])),
             card(eyebrow(T('PRÓXIMOS PASOS')), html.Div([html.Div([html.Span(f'0{k + 1}', className='obj-n'), html.Div([html.H4(T(t)), html.P(T(s))])], className='next') for k, (t, s) in enumerate(steps)]))),
        card(eyebrow(T('REPRODUCIBILIDAD')), html.H3(T('Todo lo que se reporta se puede verificar')),
             html.Div([html.Div([html.Span(icon(i), className='st-icon'), html.H4(T(t)), html.P(T(x))], className='station') for i, t, x in REPRO], className='repro'),
             html.A([icon('code', 'sm'), T('Ver el código del experimento en el Jupyter Book'), icon('arrow', 'sm')], href=BOOK + 'CODIGO_EXPERIMENTO.html', target='_blank', className='chip-link')),
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
    res = [('code', T('Código y notebooks'), T('Repositorio Telco_Churn_JBook'), REPO), ('sliders', T('Este dashboard'), T('Repositorio Dash_Telco_Churn'), DASH_REPO),
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
          N('Sin seguridad en línea: 41.9 %'), N('Mejor F1: 0.638 · Random Forest + SMOTE'), N('SMOTE mejoró 6 de 7 familias'), N('Contrato, internet y antigüedad lideran SHAP'), N('LinearSVC: 36× más rápido que SVC RBF'), N('Successive Halving: 7.1× menos tiempo en SVM')]


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
    fs.update_traces(diagonal_visible=False, showupperhalf=False, marker_size=4)
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
    for col, name, color in [('probability_native', T('Nativa'), SOFT), ('probability_sigmoid', T('Sigmoide (Platt)'), BASE), ('probability_isotonic', T('Isotónica'), HOT)]:
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
    f = go.Figure(go.Bar(x=top[metric + '_mean'], y=top.n, orientation='h', error_x={'type': 'data', 'array': top[metric + '_std'], 'color': GREY}, marker_color=highlight(top[metric + '_mean'], len(top) - 1)))
    lo = max(0, top[metric + '_mean'].min() - .06)
    bar_labels(f, top[metric + '_mean'], top.n, lo + .001)
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
                     grid(chart(T('Diversidad del algoritmo genético'), fd, T('La población pierde diversidad con las generaciones: converge hacia pocas configuraciones.'), 340),
                          chart(T('Presupuesto y coste empírico'), fb, T('Segundos por unidad externa en el equipo del experimento.'), 340, False))])


@app.callback(Output('cal-map-output', 'children'), Input('cal-family', 'value'), State('lang', 'data'))
def cal_map(family, lang='es'):
    """Cómo transforma cada recalibrador el score nativo: curva en S (Platt) frente a escalera (isotónica)."""
    set_lang(lang); e = BEST_BY_FAMILY[family]; o = oof()[lambda x: x.experiment == e]
    xcol = 'probability_native' if o.probability_native.notna().any() else 'score_positive'
    s = o.sort_values(xcol).iloc[::3]
    f = go.Figure()
    if xcol == 'probability_native': f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line={'dash': 'dash', 'color': GREY}, name=T('Sin cambio'), hoverinfo='skip'))
    for col, name, color in [('probability_sigmoid', T('Sigmoide (Platt)'), BASE), ('probability_isotonic', T('Isotónica'), HOT)]:
        f.add_trace(go.Scatter(x=s[xcol], y=s[col], mode='markers', name=name, marker={'color': color, 'size': 5, 'opacity': .55}))
    f.update_xaxes(title=T('Probabilidad nativa') if xcol == 'probability_native' else T('Margen de SVM (score)')); f.update_yaxes(title=T('Probabilidad recalibrada'), range=[0, 1])
    n_iso = int(o.groupby('outer_fold').probability_isotonic.nunique().mean())
    return html.Div([eyebrow(config_name(e)),
                     graph(f, 340), html.P(T('Cada punto es un cliente fuera de muestra (5 folds externos). Platt dibuja una curva suave; la isotónica, escalones: en promedio {n} probabilidades distintas por fold.').format(n=n_iso), className='insight')])


@app.callback(Output('eff-variants-output', 'children'), Input('eff-family', 'value'), State('lang', 'data'))
def eff_variants(family, lang='es'):
    set_lang(lang); v = comp('estandar_vs_optimizado')[lambda x: x.model == family].iloc[::-1]
    f = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[T('Aceleración al entrenar'), T('Aceleración al predecir')], horizontal_spacing=.04)
    for j, col in enumerate(['aceleración_entrenamiento', 'aceleración_inferencia'], 1):
        f.add_trace(go.Bar(x=v[col], y=v.variant, orientation='h', marker_color=highlight(v[col]), text=v[col].map('{:.2f}×'.format), textposition='outside', cliponaxis=False, showlegend=False), row=1, col=j)
        f.add_vline(x=1, line_dash='dot', line_color=GREY, row=1, col=j)
        f.update_xaxes(type='log', title_text=T('veces más rápido (escala log)'), row=1, col=j)
    t = v[::-1][['variant', 'complejidad', 'entrenamiento_s', 'inferencia_s', 'delta_f1']].rename(columns={'variant': 'Variante', 'complejidad': 'Complejidad', 'entrenamiento_s': 'Entrenamiento (s)', 'inferencia_s': 'Inferencia (s)', 'delta_f1': 'ΔF1'})
    return html.Div([graph(f, max(260, 70 + 52 * len(v)), False), table(tt(t.round(4)), page_size=6),
                     html.P(T('Valores mayores que 1 significan más rápido que la versión estándar (la primera fila); la más rápida está resaltada. ΔF1 compara con la estándar en el fold externo.'), className='insight')])


@app.callback(Output('eff-scale-output', 'children'), Input('eff-scale', 'value'), State('lang', 'data'))
def eff_scale(model, lang='es'):
    set_lang(lang); d = comp('escalamiento')[lambda x: x.model == model]
    f = make_subplots(rows=1, cols=2, subplot_titles=[T('Escalamiento en n'), T('Escalamiento en p')], horizontal_spacing=.12)
    for j, axis in enumerate(['n', 'p'], 1):
        s = d[d.axis == axis].sort_values('value')
        f.add_trace(go.Scatter(x=s.value, y=s.median_seconds, mode='lines+markers', line={'color': HOT if model == 'svm rbf' else BASE, 'width': 3}, marker={'size': 9}, showlegend=False), row=1, col=j)
        f.update_xaxes(type='log', title_text=T('n (filas de entrenamiento)') if axis == 'n' else T('p (columnas, con ruido añadido)'), row=1, col=j)
        f.update_yaxes(type='log', row=1, col=j)
    f.update_yaxes(title_text=T('Mediana de 3 repeticiones (s)'), row=1, col=1)
    x = comp('escalamiento_exponentes').set_index('model').loc[model]
    return html.Div([graph(f, 300, False), html.P(T('Pendiente log-log: {n} en n y {p} en p. Teoría en n: {t}.').format(n=f'{x.exponente_n:.2f}', p=f'{x.exponente_p:.2f}', t=x['teórico_n']), className='insight')])


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
