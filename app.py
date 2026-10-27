"""Señal perdida: dashboard interactivo del proyecto Telco Customer Churn de Laura Rivera y Natalý Cárdenas."""
import json
import os
import re
from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
from flask import send_from_directory
from sklearn.metrics import roc_curve, roc_auc_score
import analytics as a
import chatbot
from i18n import EN, T, N, set_lang, get_lang
from icons import icon

BOOK = 'https://nataly-cardenas.github.io/Telco_Churn_Entregable/'
REPO = 'https://github.com/Nataly-Cardenas/Telco_Churn_Entregable'
DASH_REPO = 'https://github.com/Nataly-Cardenas/Dash_Telco_Churn'

# ---------------------------------------------------------------- paleta de las gráficas
# Pensada para leerse sobre fondo claro y oscuro: tonos medios, sin negros ni blancos en las escalas.
PALETTE = {'teal': '#1BA7C4', 'violet': '#7C5CE0', 'coral': '#E8566C', 'amber': '#F2A93B', 'green': '#3BB57A', 'blue': '#4C7BE0', 'magenta': '#C2569A'}
CAT = list(PALETTE.values())                                    # series categóricas, en este orden
STAY, CHURN = PALETTE['teal'], PALETTE['coral']                  # permanece / abandona
LOW, MID, HIGH = PALETTE['green'], PALETTE['amber'], PALETTE['coral']   # niveles de riesgo
MODEL_COLORS = {'random_forest': PALETTE['teal'], 'xgboost': PALETTE['violet'], 'logistic': PALETTE['coral'], 'svm': PALETTE['amber'],
                'naive_bayes': PALETTE['green'], 'knn': PALETTE['blue'], 'decision_tree': PALETTE['magenta']}
RISK_SCALE = [[0, PALETTE['teal']], [.5, PALETTE['violet']], [1, PALETTE['coral']]]          # secuencial: bajo → alto abandono
DIVERGING = [[0, PALETTE['teal']], [.5, 'rgba(139,147,167,.18)'], [1, PALETTE['coral']]]       # correlaciones −1 … 1
VALUE_SCALE = [[0, 'rgba(124,92,224,.25)'], [.5, PALETTE['violet']], [1, PALETTE['teal']]]     # métricas: más alto = mejor
GRID = 'rgba(139,147,167,.18)'

NAV = [('problema', N('El problema'), 'signal'), ('datos', N('Datos'), 'chart'), ('preparacion', N('Preparación'), 'flask'), ('modelos', N('Arena de modelos'), 'trophy'),
       ('simulador', N('Simulador'), 'sliders'), ('porque', N('¿Por qué se van?'), 'search'), ('conclusiones', N('Conclusiones'), 'flag'), ('autoras', N('Autoras'), 'users')]
TITLES = {'inicio': N('Inicio'), **{k: t for k, t, _ in NAV}}
DATA_PAGES = {'datos'}

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


# ---------------------------------------------------------------- traducción de datos
def cat(v):
    """Nombre visible de una categoría del dataset (vienen en inglés)."""
    if get_lang() == 'en': return {0: 'No', 1: 'Yes'}.get(v, v) if not isinstance(v, str) else v
    return a.ES.get(v, v) if isinstance(v, str) else {0: 'No', 1: 'Sí'}.get(v, v)
def label(var): return T(a.LABELS.get(var, var))
def model_name(m): return T(a.MODELS.get(m, m))
def balance_name(b): return T(a.BALANCES.get(b, b))
def optimizer_name(o): return T(a.OPTIMIZERS.get(o, o))
def outcome(d): return d.target.map({0: T('Permanece'), 1: T('Abandona')})
def outcome_colors(): return {T('Permanece'): STAY, T('Abandona'): CHURN}


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


def table(d, id=None, **kw):
    args = {'id': id} if id else {}
    return dash_table.DataTable(data=d.round(4).replace({np.nan: None}).to_dict('records'), columns=[{'name': c, 'id': c} for c in d.columns],
                                page_size=kw.pop('page_size', 10), sort_action='native', filter_action=kw.pop('filter_action', 'native'), style_table={'overflowX': 'auto'},
                                style_header={'backgroundColor': 'var(--table-head)', 'color': 'var(--text)', 'border': 'none', 'fontWeight': '600'},
                                style_cell={'backgroundColor': 'var(--table-cell)', 'color': 'var(--soft)', 'border': 'none', 'borderBottom': '1px solid var(--line)', 'padding': '10px',
                                            'textAlign': 'left', 'fontFamily': 'Inter, sans-serif', 'fontSize': 12, 'maxWidth': 320, 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                                style_filter={'backgroundColor': 'var(--table-head)', 'color': 'var(--text)'}, **args, **kw)


def style(f, height=340, legend=True, margin=None):
    # Los colores de texto y rejilla se ajustan al tema desde CSS (assets/style.css, sección "gráficas").
    f.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter, sans-serif', 'color': '#8792a8', 'size': 12}, colorway=CAT,
                    margin=margin or {'l': 20, 'r': 20, 't': 30, 'b': 30}, height=height, showlegend=legend,
                    legend={'orientation': 'h', 'y': 1.13, 'title_text': '', 'bgcolor': 'rgba(0,0,0,0)'},
                    hoverlabel={'bgcolor': '#0b1426', 'bordercolor': PALETTE['teal'], 'font': {'family': 'Inter', 'color': '#eaf2ff'}})
    f.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID); f.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return f


def graph(f, height=340, legend=True, margin=None):
    return dcc.Graph(figure=style(f, height, legend, margin), style={'height': height, 'width': '100%'},
                     config={'displaylogo': False, 'responsive': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


def chart(title, f, insight='', height=340, legend=True):
    return card(html.Div([html.H3(title), html.Span(T('en vivo'), className='live')], className='card-head'), graph(f, height, legend),
                html.P(insight, className='insight') if insight else None)


def page_head(key, lead, chapter=None):
    chapters = [chapter] if isinstance(chapter, str) else chapter or []
    book = html.Div([html.Span(T('Capítulos del libro en esta sección:'), className='muted'), *[src(c) for c in chapters]], className='book-map') if chapters else None
    return html.Div([eyebrow('TELCO CUSTOMER CHURN · ' + T(TITLES[key]).upper()), html.H1(T(TITLES[key])), html.P(lead, className='lead'), book], className='page-head')


def tabs(id, items):
    return dcc.Tabs([dcc.Tab(children=html.Div(content, className='tab-body'), label=lbl, value=str(i), className='tab', selected_className='tab on')
                     for i, (lbl, content) in enumerate(items)], value='0', id=id, className='tabs', parent_className='tabs-wrap')


# Capítulos del Jupyter Book: cada parte del dash indica de qué capítulo y sección viene.
BOOK_CHAPTERS = {'intro': (N('Introducción'), 'intro'), 'EDA_': (N('Análisis exploratorio (EDA)'), 'eda'), 'PREPROCESAMIENTO': (N('Preprocesamiento'), 'pre'),
                 'MODELADO_AVANZADO': (N('Modelado avanzado'), 'mod'), 'paper': ('Paper', 'paper')}


def src(chapter, section=None, anchor=''):
    """Etiqueta de procedencia: capítulo y sección del Jupyter Book, con enlace directo."""
    name, cls = BOOK_CHAPTERS[chapter]
    return html.A([icon('book', 'sm'), html.Span('Jupyter Book · ' + T(name), className='src-ch'), html.Span(' › ' + T(section)) if section else None, icon('arrow', 'sm')],
                  href=BOOK + chapter + '.html' + ('#' + anchor if anchor else ''), target='_blank', className=f'src src-{cls}', title=T('Abrir esta sección en el Jupyter Book'))


def dash_only(based_on=None):
    """Etiqueta para herramientas creadas para el dash (no existen como tal en el libro)."""
    return html.Span([icon('sliders', 'sm'), T('Herramienta del dash'), html.Span(' · ' + T(based_on)) if based_on else None], className='src src-dash')


def section(title, *tags, sub=None):
    return html.Div([html.Div([html.H2(title), *[t for t in tags if t]], className='section-row'), html.P(sub, className='muted') if sub else None], className='section-head')


def empty(): return note(T('Sin clientes en este segmento'), T('Amplía los filtros para volver a calcular las visualizaciones.'))
def scope(d): return html.Div([html.Span(className='live-dot'), ' ' + T('{n} clientes en la selección · análisis descriptivo').format(n=f'{len(d):,}')], className='scope')
def details(summary, *children): return html.Details([html.Summary(summary), *children], className='details')


# ---------------------------------------------------------------- portada
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
                html.Div([dcc.Link([T('Entrar al dashboard'), ' →'], href='#problema', className='btn primary'),
                          html.A([T('Jupyter Book'), icon('arrow', 'sm')], href=BOOK, target='_blank', className='btn ghost')], className='cover-cta'),
                html.Div([icon('cursor'), html.Span([T('Los puntos rojos son clientes a punto de irse: '), html.B(T('pasa el cursor (o el dedo) sobre ellos para retenerlos.'))]),
                          html.Span([T('Perdidos: '), html.B('0', id='cover-lost'), ' · ', T('Retenidos por ti: '), html.B('0', id='cover-saved')], className='score')], className='cover-hint'),
            ], className='cover-text'),
            phone()], className='cover-inner'),
        html.Div('Laura S. Rivera Figueredo · Natalý A. Cárdenas Izaquita', className='cover-foot')], className='cover')


# ---------------------------------------------------------------- 1. el problema
def people_row():
    leaving = {2, 7, 11, 16, 19}   # 5 de 20 ≈ 26.5 % de abandono
    return html.Div([html.Span(className='person' + (' leaving' if i in leaving else ''), style={'animationDelay': f'{(i % 5) * .9}s'}) for i in range(20)], className='people')


FLIP = [('target', N('Clasificación binaria'), N('Estimar para cada cliente la probabilidad P(abandono | x) y separar a quienes se van (1) de quienes permanecen (0).')),
        ('scale', N('Desbalance de clases'), N('Solo el 26.5 % abandona. SMOTE y ADASYN crean ejemplos sintéticos de la minoría para que el modelo no la ignore.')),
        ('block', N('Fuga de información'), N('Imputar, codificar, escalar y balancear solo con datos de entrenamiento; si no, las métricas mienten.')),
        ('tree', N('Siete modelos'), N('Regresión logística (base), KNN, Naive Bayes, árbol, Random Forest, XGBoost y SVM compiten bajo las mismas reglas.')),
        ('repeat', N('Validación anidada'), N('Un bucle interno elige hiperparámetros y uno externo mide el desempeño con datos que no participaron en la elección.')),
        ('sigma', N('F1 y AUC'), N('F1 = 2TP / (2TP + FP + FN) equilibra detectar abandonos y evitar falsas alertas; AUC mide qué tan bien ordena el riesgo.')),
        ('bulb', N('SHAP y LIME'), N('Explican qué variables empujan cada predicción. Describen al modelo, no causas del abandono.'))]
OBJECTIVES = [(N('Describir'), N('Caracterizar los datos, su calidad y el desbalance del abandono.')), (N('Reservar'), N('Separar 20 % del test antes de mirar los datos.')),
              (N('Perfilar'), N('Encontrar los segmentos con más cancelaciones.')), (N('Preparar'), N('Transformar sin fuga de información.')),
              (N('Referenciar'), N('Entrenar una regresión logística como modelo base.')), (N('Comparar'), N('112 configuraciones con validación cruzada anidada.')),
              (N('Explicar'), N('Identificar con SHAP y LIME qué variables influyen.')), (N('Compartir'), N('Publicar libro, paper y este dashboard.'))]
GROUPS = [('card', N('Cuenta'), ['Contract', 'PaymentMethod', 'PaperlessBilling', 'tenure', 'MonthlyCharges', 'TotalCharges']),
          ('wifi', N('Servicios'), ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']),
          ('users', N('Demografía'), ['gender', 'SeniorCitizen', 'Partner', 'Dependents'])]


def problema(d):
    return html.Div([
        page_head('problema', T('Perder un cliente cuesta más que conservarlo. Este proyecto busca anticipar quién está por cancelar su servicio para actuar a tiempo.'), ['intro', 'EDA_']),
        html.Div([html.Div([src('intro', N('Introducción y contexto del problema'), 'introduccion'), eyebrow(T('EL RETO')), html.H2([T('1 de cada 4 clientes '), html.Span(T('se desconecta'), className='txt-coral')]),
                            html.P(T('El conjunto Telco Customer Churn de IBM reúne servicios contratados, datos de cuenta y perfil demográfico de cada cliente, '
                                     'junto con si canceló en el último mes. Dos dificultades: los que se van son minoría, y los datos son una foto de un momento.')),
                            people_row()], className='hero-text'),
                  kpis(kpi(T('CLIENTES'), 7043, T('una fila por cliente')), kpi(T('ABANDONOS'), 1869, T('cancelaron en el último mes'), 'coral'),
                       kpi(T('TASA DE ABANDONO'), 26.5, T('la clase minoritaria'), 'coral', 1, ' %'), kpi(T('VARIABLES'), 19, T('predictoras candidatas')), cls='k2')], className='hero'),
        section(T('¿Qué buscamos?'), src('intro', N('Objetivo general y objetivos específicos'), 'objetivo-general')),
        card(eyebrow(T('OBJETIVO GENERAL')), html.P(T('Identificar a los clientes con mayor riesgo de abandonar el servicio y los factores asociados, comparando de forma reproducible '
                                                     'modelos de clasificación que detecten la mayor cantidad de clientes en riesgo sin generar un exceso de falsas alertas.'), className='big-text'), cls='glow'),
        html.Div([html.Div([html.Span(f'{i + 1:02d}', className='obj-n'), html.H4(T(t)), html.P(T(s))], className='obj') for i, (t, s) in enumerate(OBJECTIVES)], className='objectives'),
        section(T('Marco teórico'), src('intro', N('Marco teórico'), 'marco-teorico'), sub=T('Pasa el cursor (o toca) cada tarjeta para voltearla.')),
        html.Div([html.Div(html.Div([html.Div([html.Span(icon(i), className='flip-icon'), html.H4(T(t))], className='flip-front'), html.Div([html.H4(T(t)), html.P(T(s))], className='flip-back')],
                                    className='flip-inner'), className='flip', tabIndex=0) for i, t, s in FLIP], className='flips'),
        section(T('Análisis bivariado: abandono según cada variable'), src('EDA_', N('4. Análisis bivariado'), 'analisis-bivariado'),
                src('intro', N('Descripción de las variables'), 'descripcion-de-las-variables-del-proyecto')),
        card(html.P(T('Elige una variable para ver cómo cambia la tasa de abandono entre sus categorías (conjunto de entrenamiento).'), className='muted'),
             html.Div([html.Span([icon(i), T(g)], className=f'chip-group grp-{k}') for k, (i, g, _) in enumerate(GROUPS)], className='chip-groups'),
             dcc.RadioItems([{'label': html.Span(label(v), className=f'grp-{k}'), 'value': v} for k, (_, _, vs) in enumerate(GROUPS) for v in vs], 'Contract', id='var-chip', className='chips', inline=True),
             html.Div(id='var-chip-output'))])


# ---------------------------------------------------------------- 2. datos
def datos(d):
    if d.empty: return html.Div([page_head('datos', T('Explora el conjunto de datos con los filtros superiores.'), 'EDA_'), empty()])
    audit = pd.DataFrame({T('Control'): [T('Clientes'), T('Identificadores repetidos'), T('Filas repetidas'), T('TotalCharges vacío'), T('Vacío con tenure ≠ 0')],
                          T('Resultado'): [len(d), int(d.customerID.duplicated().sum()), int(d[a.RAW.columns[:21]].duplicated().sum()), int(d.TotalCharges.isna().sum()),
                                           int((d.TotalCharges.isna() & (d.tenure != 0)).sum())]})
    split = go.Figure(go.Pie(labels=[T('Entrenamiento'), T('Test histórico')], values=[5634, 1409], hole=.72, marker_colors=[STAY, PALETTE['violet']], textinfo='percent', sort=False))
    split.add_annotation(text='<b>80 / 20</b><br>' + T('estratificado'), showarrow=False, font_size=16)
    dictionary = pd.DataFrame([{T('Variable'): c, T('Significado'): label(c) if c in a.LABELS else T('Identificador') if c == 'customerID' else T('Abandono'),
                                T('Tipo'): T('Numérica') if c in a.NUMERIC else T('Categórica'), T('Rol'): T('Objetivo') if c == 'Churn' else T('No predictor') if c == 'customerID' else T('Predictor')}
                               for c in a.RAW.columns[:21]])
    calidad = [html.Div([src('EDA_', N('1. ETL'), 'etl'), src('EDA_', N('2. División train / test'), 'division-train-test'), src('EDA_', N('6. Valores faltantes'), 'valores-faltantes')], className='srcs'),
               kpis(kpi(T('FILAS'), 7043, T('una por cliente')), kpi(T('ENTRENAMIENTO'), 5634, T('80 % · aquí se hace el EDA')), kpi(T('TEST'), 1409, T('20 % · reservado')),
                    kpi(T('VACÍOS'), 11, T('TotalCharges con tenure = 0'), 'amber')),
               grid(chart(T('Partición antes de mirar los datos'), split, T('El test se separa primero para que ninguna decisión se tome con él.'), 300, False),
                    card(html.H3(T('Auditoría de la selección')), table(audit, filter_action='none'), html.P(T('Los 11 vacíos son clientes nuevos sin facturación acumulada: '
                         'se reemplazan por 0 en el preprocesamiento, no por un promedio ficticio.'), className='insight'))),
               details([icon('book'), T('Diccionario de variables')], table(dictionary, page_size=21))]
    distribuciones = [html.Div([src('EDA_', N('3. Análisis univariado'), 'analisis-univariado'), src('EDA_', N('3.4 Outliers y sesgos'), 'outliers-y-sesgos')], className='srcs'), card(field(T('Variable'), dropdown('uni-var', a.NUMERIC + a.CATEGORICAL, 'tenure', {v: label(v) for v in a.LABELS})), cls='controls-card'), html.Div(id='uni-output')]
    p1, p2, p3 = T('Contrato'), T('Internet'), T('Pago')
    sun = d.assign(**{p1: d.Contract.map(cat), p2: d.InternetService.map(cat), p3: d.PaymentMethod.map(cat)}).groupby([p1, p2, p3]).target.agg(['size', 'mean']).reset_index()
    sb = px.sunburst(sun, path=[p1, p2, p3], values='size', color='mean', color_continuous_scale=RISK_SCALE, range_color=[0, .7], labels={'mean': T('Abandono'), 'size': T('Clientes')})
    sb.update_traces(hovertemplate='<b>%{label}</b><br>' + T('Clientes') + ': %{value}<br>' + T('Abandono') + ': %{color:.1%}<extra></extra>', insidetextorientation='radial')
    segmentos = [html.Div([src('EDA_', N('4. Análisis bivariado'), 'analisis-bivariado'), src('EDA_', N('Categorías con mayores tasas de churn'), 'categorias-con-mayores-tasas-de-churn')], className='srcs'), grid(chart(T('Mapa de riesgo · haz clic para profundizar'), sb, T('Contrato → internet → método de pago. El color es la tasa de abandono: el coral marca '
                            'los perfiles más frágiles, como mes a mes + fibra óptica + cheque electrónico.'), 470, False), tenure_race(d)),
                 card(field(T('Comparar abandono por'), dropdown('bi-var', a.CATEGORICAL + a.NUMERIC, 'PaymentMethod', {v: label(v) for v in a.LABELS})), cls='controls-card'), html.Div(id='bi-output')]
    corr = d[a.NUMERIC + ['target']].rename(columns={**{v: label(v) for v in a.NUMERIC}, 'target': T('Abandono')}).corr(method='spearman')
    relaciones = [html.Div([src('EDA_', N('5. Análisis multivariado'), 'analisis-multivariado')], className='srcs'), grid(chart(T('Correlaciones de Spearman'), px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale=DIVERGING),
                             T('Antigüedad y cargo acumulado comparten mucha información (ρ ≈ 0.89): por eso se crea el cargo promedio mensual en lugar de borrar una.'), 380, False),
                       card(html.H3(T('Dos variables a la vez')), grid(field(T('Eje X'), dropdown('scatter-x', a.NUMERIC, 'tenure', {v: label(v) for v in a.NUMERIC})),
                                                                       field(T('Eje Y'), dropdown('scatter-y', a.NUMERIC, 'MonthlyCharges', {v: label(v) for v in a.NUMERIC}))),
                            html.Div(id='scatter-output'), cls='controls-card'))]
    explorador = [html.Div([dash_only(N('datos originales de Kaggle'))], className='srcs'), card(field(T('Columnas visibles'), dropdown('explorer-cols', list(a.RAW.columns[:21]), ['customerID', 'Contract', 'InternetService', 'tenure', 'MonthlyCharges', 'Churn'], {}, multi=True)),
                       html.Div(id='explorer-table'),
                       html.Div([html.Button([icon('download'), T('Descargar CSV')], id='export', className='btn primary', n_clicks=0),
                                 html.P(T('Descarga un solo archivo con la selección actual de los filtros superiores y las columnas elegidas.'), className='muted')], className='export-row'), cls='controls-card')]
    return html.Div([page_head('datos', T('Del archivo original a los patrones de abandono. Los filtros superiores recalculan todo lo que ves aquí.'), 'EDA_'), scope(d),
                     kpis(kpi(T('CLIENTES'), len(d), T('en la selección')), kpi(T('ABANDONO'), round(d.target.mean() * 100, 2), T('{n} clientes').format(n=f'{int(d.target.sum()):,}'), 'coral', 2, ' %'),
                          kpi(T('ANTIGÜEDAD MEDIA'), round(d.tenure.mean(), 1), T('meses'), '', 1), kpi(T('CARGO MENSUAL'), round(d.MonthlyCharges.mean(), 2), T('promedio'), '', 2, '', '$')),
                     tabs('datos-tabs', [(T('1–2 · Calidad y ETL'), calidad), (T('3 · Análisis univariado'), distribuciones), (T('4 · Análisis bivariado'), segmentos),
                                         (T('5 · Análisis multivariado'), relaciones), (T('Explorador de datos'), explorador)])])


def tenure_race(d):
    """Animación: la tasa de abandono por contrato a medida que crece la antigüedad."""
    bins = [-1, 6, 12, 24, 36, 48, 60, 72]; names = ['0–6', '7–12', '13–24', '25–36', '37–48', '49–60', '61–72']
    col, frame = T('Contrato'), T('Antigüedad (meses)')
    g = d.assign(**{frame: pd.cut(d.tenure, bins, labels=names), col: d.Contract.map(cat)}).groupby([frame, col], observed=False).target.agg(tasa='mean', n='size').reset_index().fillna({'tasa': 0})
    g['txt'] = g.tasa.map('{:.0%}'.format)
    order = [cat('Month-to-month'), cat('One year'), cat('Two year')]
    f = px.bar(g, x=col, y='tasa', color=col, animation_frame=frame, range_y=[0, .75], text='txt', custom_data=['n'], category_orders={col: order},
               color_discrete_map=dict(zip(order, [CHURN, PALETTE['violet'], STAY])))
    f.update_traces(hovertemplate='%{x}<br>' + T('Abandono') + ': %{y:.1%}<br>n=%{customdata[0]}<extra></extra>', textposition='outside')
    f.update_yaxes(tickformat='.0%', title=T('Tasa de abandono')); f.update_xaxes(title='')
    f.layout.updatemenus[0].buttons[0].args[1].update(frame={'duration': 900, 'redraw': True}, transition={'duration': 500})
    return chart(T('Dale play: el riesgo cae con la antigüedad'), f, T('Cada fotograma es un tramo de antigüedad. Los clientes nuevos con contrato mes a mes son los más '
                 'propensos a irse; con los años la tasa se desploma.'), 470, False)


# ---------------------------------------------------------------- 3. preparación
STEPS = [('target', N('Objetivo'), N('Churn Yes/No → 1/0. customerID solo para control.')), ('shield', N('Limpieza'), N('TotalCharges a número; vacíos con tenure 0 → 0.')),
         ('layers', N('Categorías'), N('“Sin internet / sin teléfono” → “No”.')), ('sigma', N('Nueva variable'), N('AvgChargePerMonth = TotalCharges / tenure.')),
         ('table', N('Codificar y escalar'), N('One-Hot + StandardScaler ajustados en cada fold.')), ('scale', N('Balancear'), N('Ninguno, pesos, SMOTE o ADASYN; solo en train.')),
         ('network', N('Modelo'), N('Se entrena y se evalúa en el fold reservado.'))]


def preparacion(d):
    col = T('Variable retirada')
    abl = pd.DataFrame({col: ['gender', 'PhoneService', T('Ambas'), 'PhoneService + MultipleLines', 'AvgChargePerMonth'], 'ΔAUC': [.0002, .0002, .0005, -.0006, .0006]})
    fa = px.bar(abl, x='ΔAUC', y=col, orientation='h', color='ΔAUC', color_continuous_scale=[[0, CHURN], [.5, 'rgba(139,147,167,.4)'], [1, LOW]], range_color=[-.001, .001])
    fa.update_coloraxes(showscale=False); fa.update_xaxes(tickformat='.4f', title=T('Cambio en AUC al retirar')); fa.update_yaxes(title='')
    m = T('Métrica')
    base = pd.DataFrame({m: ['AUC', 'Recall', 'Precision', 'Accuracy'], 'v': [.8419, .7861, .5052, .7388]})
    fb = px.bar(base, x='v', y=m, orientation='h', text=base.v.map('{:.3f}'.format), color=m, color_discrete_sequence=[STAY, CHURN, PALETTE['violet'], LOW])
    fb.update_xaxes(range=[0, 1], title=''); fb.update_yaxes(title='')
    return html.Div([page_head('preparacion', T('Siete estaciones por las que pasa cada cliente antes de llegar al modelo, todas ajustadas sin espiar el test.'), 'PREPROCESAMIENTO'),
        section(T('Pasos de preprocesamiento'), src('PREPROCESAMIENTO', N('4. Pasos de preprocesamiento definidos a partir del EDA'), 'pasos-de-preprocesamiento-definidos-a-partir-del-eda')),
        card(html.H3(T('El pipeline en movimiento')), html.Div([html.Div(className='flow-line'), *[html.Div([html.Span(icon(i), className='st-icon'), html.H4(T(t)), html.P(T(s))], className='station',
             style={'animationDelay': f'{k * .15}s'}) for k, (i, t, s) in enumerate(STEPS)]], className='pipeline')),
        section(T('Validación cruzada en entrenamiento'), src('PREPROCESAMIENTO', N('4.6 Validación cruzada en train'), 'validacion-cruzada-en-train-protocolo-antes-de-tocar-test')),
        kpis(kpi(T('AUC IMPUTANDO CON 0'), .8459, T('CV en entrenamiento'), '', 4), kpi(T('AUC CON MEDIANA'), .8460, T('diferencia despreciable'), '', 4),
             kpi(T('ETIQUETAS PERMUTADAS'), .49, T('control: azar'), 'coral', 2), kpi(T('COLUMNAS FINALES'), 24, T('tras codificar'))),
        section(T('Modelo base y ablación'), src('PREPROCESAMIENTO', N('7. Evaluación final en test'), 'evaluacion-final-en-test-una-sola-vez'),
                src('PREPROCESAMIENTO', N('4.7 Ablación por CV'), 'ablacion-por-cv-aportan-gender-phoneservice-y-avgchargepermonth')),
        grid(chart(T('Modelo base: regresión logística en test'), fb, T('Detecta casi 8 de cada 10 abandonos, pero la mitad de sus alertas son falsas: hay margen de mejora.'), 300, False),
             chart(T('Ablación: ¿qué pasa si quito una variable?'), fa, T('Retirar gender o PhoneService casi no cambia el AUC; las diferencias están dentro del ruido entre folds.'), 300, False)),
        grid(note(T('¿Por qué SMOTE solo en entrenamiento?'), T('Si se balanceara antes de dividir, ejemplos sintéticos derivados de clientes de validación contaminarían el entrenamiento.')),
             note(T('Limitación honesta'), T('SMOTE y ADASYN se aplicaron después del One-Hot y pueden producir indicadores fraccionarios; se reporta tal cual.'), 'amber')),
        chapters('PREPROCESAMIENTO', r'4\.|7\.4|8\.')])


# ---------------------------------------------------------------- 4. modelos
METRICS = {'f1_positive': N('F1 positivo'), 'recall_positive': 'Recall', 'precision_positive': 'Precision', 'roc_auc': 'ROC-AUC', 'accuracy': 'Accuracy'}


def modelos(d):
    s = a.SUMMARY.assign(Modelo=a.SUMMARY.model.map(model_name), Balanceo=a.SUMMARY.balance.map(balance_name), Optimizador=a.SUMMARY.optimizer.map(optimizer_name))
    s = s.rename(columns={'Modelo': T('Modelo'), 'Balanceo': T('Balanceo'), 'Optimizador': T('Optimizador')})
    f3 = px.scatter_3d(s, x='recall_positive_mean', y='precision_positive_mean', z='roc_auc_mean', color=T('Modelo'), size='f1_positive_mean', size_max=16, opacity=.85,
                       color_discrete_map={model_name(k): v for k, v in MODEL_COLORS.items()}, hover_data={T('Balanceo'): True, T('Optimizador'): True, 'f1_positive_mean': ':.3f'},
                       labels={'recall_positive_mean': 'Recall', 'precision_positive_mean': 'Precision', 'roc_auc_mean': 'AUC', 'f1_positive_mean': 'F1'})
    axis = {'gridcolor': 'rgba(139,147,167,.35)', 'showbackground': False, 'zerolinecolor': 'rgba(139,147,167,.35)', 'color': '#8792a8'}
    f3.update_layout(scene={'bgcolor': 'rgba(0,0,0,0)', 'xaxis': axis, 'yaxis': axis, 'zaxis': axis, 'camera': {'eye': {'x': 1.6, 'y': 1.5, 'z': .9}}})
    ranking = [html.Div([src('MODELADO_AVANZADO', N('1. Modelos implementados y resultados obtenidos'), 'modelos-implementados-y-resultados-obtenidos')], className='srcs'),
               card(html.Div([field(T('Familia'), dropdown('model-family', ['Todas'] + list(a.MODELS), 'Todas', {'Todas': T('Todas'), **{k: model_name(k) for k in a.MODELS}})),
                              field(T('Balanceo'), dropdown('model-balance', ['Todos'] + list(a.BALANCES), 'Todos', {'Todos': T('Todos'), **{k: balance_name(k) for k in a.BALANCES}})),
                              field(T('Métrica'), dropdown('model-metric', list(METRICS), 'f1_positive', {k: T(v) for k, v in METRICS.items()}))], className='controls'), cls='controls-card'),
               html.Div(id='models-output'),
               chart(T('Galaxia de 112 configuraciones · gira con el mouse'), f3, T('Cada punto es una configuración; el tamaño es su F1. Las más grandes y altas combinan buen recall, precisión y AUC.'), 560)]
    carrera = [html.Div([src('MODELADO_AVANZADO', N('2. Comparación de métodos de optimización'), 'comparacion-de-metodos-de-optimizacion')], className='srcs'), card(field(T('Familia'), dropdown('opt-family', list(a.MODELS), 'random_forest', {k: model_name(k) for k in a.MODELS})), cls='controls-card'), html.Div(id='opt-output'),
               note(T('Sin ganador universal'), T('Con 24 evaluaciones los cuatro buscadores terminan prácticamente empatados; las diferencias solo aparecen con presupuestos bajos.'))]
    calib = [html.Div([src('MODELADO_AVANZADO', N('1.8 Síntesis de los resultados de calibración'), 'sintesis-de-los-resultados-de-calibracion')], className='srcs'), grid(card(html.H3(T('Probabilidades que dicen la verdad')), md(T('El balanceo distorsiona las probabilidades. La **calibración sigmoide** (Platt) se ajustó dentro del '
                       'entrenamiento y redujo **Brier** y **ECE** a la vez en **80 de 96** configuraciones.\n\n- **Brier**: error cuadrático de la probabilidad (menor es mejor).\n'
                       '- **ECE**: distancia entre la confianza y la frecuencia real.'))),
                  card(field(T('Familia'), dropdown('cal-family', list(a.MODELS), 'random_forest', {k: model_name(k) for k in a.MODELS})), html.Div(id='cal-output'), cls='controls-card'))]
    return html.Div([page_head('modelos', T('Siete familias × cuatro balanceos × cuatro buscadores = 112 configuraciones compitiendo bajo las mismas reglas.'), 'MODELADO_AVANZADO'),
                     html.Div([podium(), kpis(kpi(T('CONFIGURACIONES'), 112, '7 × 4 × 4'), kpi(T('UNIDADES EXTERNAS'), 1680, T('112 × 3 semillas × 5 folds')),
                                              kpi('FOLDS', '5 × 3', T('externos × internos')), kpi(T('PRESUPUESTO'), 24, T('evaluaciones por búsqueda')), cls='k2')], className='hero'),
                     tabs('model-tabs', [(T('1 · Ranking de modelos'), ranking), (T('1.8 · Calibración'), calib), (T('2 · Carrera de optimizadores'), carrera), (T('3 · Comparación estadística'), estadistica())])])


def podium():
    p = [('2', 'XGBoost', 'SMOTE · Random', .638, 'second'), ('1', 'Random Forest', 'SMOTE · TPE', .638, 'first'), ('3', T('Reg. logística'), 'SMOTE · Grid', .634, 'third')]
    return card(eyebrow(T('PODIO · F1 MEDIO EXTERNO')), html.Div([html.Div([html.Span(m, className='medal'), html.Strong(n), html.Small(c), html.Div(counter(v, 3), className='bar-podium')],
                className='pod ' + k) for m, n, c, v, k in p], className='podium'), html.P(T('Empate técnico: la diferencia entre RF y XGBoost es 0.0004, menor que la variación entre folds.'), className='insight'))


def estadistica():
    st = a.DATA / 'estadistica'; f = json.loads((st / 'friedman.json').read_text(encoding='utf-8'))
    mod = T('Modelo')
    test = pd.DataFrame({mod: ['Random Forest', 'XGBoost', T('Reg. logística')], 'Accuracy': [.768, .769, .747], 'Precision': [.547, .546, .515], 'Recall': [.735, .765, .794],
                         'F1': [.627, .637, .625], 'AUC': [.844, .842, .842]})
    fr = px.line_polar(test.melt(id_vars=mod, var_name=T('Métrica'), value_name='v'), r='v', theta=T('Métrica'), color=mod, line_close=True, range_r=[.45, .85],
                       color_discrete_sequence=[MODEL_COLORS['random_forest'], MODEL_COLORS['xgboost'], MODEL_COLORS['logistic']])
    fr.update_traces(fill='toself', opacity=.55); fr.update_layout(polar={'bgcolor': 'rgba(0,0,0,0)', 'radialaxis': {'gridcolor': GRID}, 'angularaxis': {'gridcolor': GRID}})
    return [html.Div([src('MODELADO_AVANZADO', N('3. Resultados finales y comparación estadística'), 'resultados-finales-comparacion-estadistica-e-interpretabilidad')], className='srcs'),
            kpis(kpi('FRIEDMAN χ²', round(f['statistic'], 2), f'Kendall W = {f["kendall_W"]:.2f}', '', 2), kpi(T('PARES NEMENYI < 0.05'), 31, T('de 6,216 · exploratorio')),
                 kpi(T('p DELONG (HOLM)'), 1.0, T('los tres pares'), '', 1), kpi('ΔAUC RF − XGB', .0017, T('IC [−0.0034, 0.0070]'), '', 4)),
            grid(chart(T('Finalistas en el test histórico'), fr, T('Los tres finalistas se superponen: la regresión logística detecta más (Recall) y los ensambles aciertan más sus alertas (Precision).'), 400),
                 card(html.H3(T('¿Hay un ganador?')), md(T('**No de forma demostrable.** Friedman detecta diferencias entre las 112 configuraciones, pero Nemenyi no separa a las mejores. '
                      'DeLong con corrección de Holm da *p* = 1.0 en los tres pares y todos los intervalos bootstrap del AUC incluyen el cero.\n\n'
                      'No detectar diferencias **no** prueba que sean equivalentes, y el test ya había sido expuesto: es evidencia suplementaria.')))),
            card(html.H3(T('Comparaciones DeLong')), table(pd.read_csv(st / 'delong_holm.csv'), filter_action='none'))]


# ---------------------------------------------------------------- 5. simulador
PAY = list(a.RAW.PaymentMethod.unique())
GAUGE_M = {'l': 34, 'r': 34, 't': 50, 'b': 10}


def simulador(d):
    perfil = [html.Div([dash_only(N('basada en el análisis bivariado')), src('EDA_', N('4. Análisis bivariado'), 'analisis-bivariado')], className='srcs'),
              grid(card(html.H3(T('Arma un cliente')), field(T('Contrato'), dropdown('sim-contract', ['Month-to-month', 'One year', 'Two year'], 'Month-to-month')),
                        field(T('Internet'), dropdown('sim-internet', ['Fiber optic', 'DSL', 'No'], 'Fiber optic')), field(T('Método de pago'), dropdown('sim-payment', PAY, 'Electronic check')),
                        field(T('Antigüedad (meses)'), dcc.Slider(0, 72, 1, value=4, marks={0: '0', 24: '24', 48: '48', 72: '72'}, id='sim-tenure', tooltip={'placement': 'bottom'})),
                        field(T('Cargo mensual ($)'), dcc.Slider(18, 119, 1, value=85, marks={20: '20', 60: '60', 100: '100'}, id='sim-monthly', tooltip={'placement': 'bottom'})), cls='controls-card'),
                   html.Div(id='sim-output')),
              note(T('¿De dónde sale este riesgo?'), T('Es la tasa de abandono observada en clientes de entrenamiento con un perfil parecido, no una predicción del modelo. '
                   'Sirve para explorar patrones; para la predicción del modelo usa la pestaña de clientes reales.'))]
    reales = [html.Div([dash_only(N('con las predicciones de los finalistas')), src('MODELADO_AVANZADO', N('3. Modelos finales en el test histórico'), 'modelos-finales-y-procedencia-de-las-comparaciones-dirigidas')], className='srcs'), card(html.Div([field(T('Cliente del test histórico'), dcc.Dropdown([{'label': a.RAW.customerID[i], 'value': i} for i in a.TEST_IDS], int(a.TEST_IDS[0]), id='real-id', clearable=False)),
                             html.Button([icon('shuffle'), T('Cliente al azar')], id='real-random', className='btn ghost', n_clicks=0)], className='controls'),
                    html.Div(id='real-output'), cls='controls-card')]
    umbral = [html.Div([dash_only(N('con las predicciones de los finalistas')), src('MODELADO_AVANZADO', N('3. Resultados finales'), 'resultados-finales-comparacion-estadistica-e-interpretabilidad')], className='srcs'), card(html.Div([field(T('Modelo finalista'), dropdown('threshold-model', a.FINALISTS, a.FINALISTS[0], {k: T(v) for k, v in a.FINALIST_NAMES.items()})),
                             field(T('Umbral de alerta'), dcc.Slider(.05, .95, .01, value=.5, marks={.1: '0.10', .3: '0.30', .5: '0.50', .7: '0.70', .9: '0.90'}, id='threshold', tooltip={'always_visible': True}))],
                            className='controls'), html.Div(id='threshold-output'), cls='controls-card'),
              note(T('Decisión de negocio'), T('Bajar el umbral detecta más clientes en riesgo pero dispara más falsas alertas. Elegirlo bien requiere conocer el costo de cada contacto y de cada cliente perdido.'))]
    return html.Div([page_head('simulador', T('Juega con perfiles, revisa clientes reales y decide dónde poner la alarma de retención.'), ['EDA_', 'MODELADO_AVANZADO']),
                     tabs('sim-tabs', [(T('Perfil de riesgo'), perfil), (T('Clientes reales'), reales), (T('Umbral de alerta'), umbral)])])


def gauge(value, title, color):
    return go.Figure(go.Indicator(mode='gauge+number', value=value * 100, number={'suffix': ' %', 'font': {'size': 34}}, title={'text': title, 'font': {'size': 13}},
                                  gauge={'axis': {'range': [0, 100], 'tickcolor': '#8792a8'}, 'bar': {'color': color, 'thickness': .28}, 'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                                         'steps': [{'range': [0, 35], 'color': 'rgba(59,181,122,.14)'}, {'range': [35, 60], 'color': 'rgba(242,169,59,.14)'}, {'range': [60, 100], 'color': 'rgba(232,86,108,.16)'}]}))


def level(p): return ('alto', HIGH) if p >= .6 else ('medio', MID) if p >= .35 else ('bajo', LOW)


def signal(p):
    lit = max(1, round((1 - p) * 5))
    return html.Div([html.B(className='lit' if i < lit else '', style={'height': f'{12 + i * 9}px'}) for i in range(5)], className='signal ' + level(p)[0])


# ---------------------------------------------------------------- 6. por qué se van
def porque(d):
    t = a.RAW.loc[a.TRAIN_IDS]
    drivers = [('file', T('Contrato'), T('Mes a mes'), t[t.Contract == 'Month-to-month'].target.mean(), T('vs 2.9 % con dos años')),
               ('globe', 'Internet', T('Fibra óptica'), t[t.InternetService == 'Fiber optic'].target.mean(), T('vs 7.3 % sin internet')),
               ('hourglass', T('Antigüedad'), T('Primer año'), t[t.tenure <= 12].target.mean(), T('vs {p} después de 4 años').format(p=f'{t[t.tenure > 48].target.mean():.1%}'))]
    return html.Div([page_head('porque', T('Lo que los modelos aprendieron: las variables que empujan el riesgo, y cómo explicaron aciertos y errores concretos.'), ['EDA_', 'MODELADO_AVANZADO']),
        section(T('Factores asociados al abandono'), src('EDA_', N('4.2 Variable objetivo vs variables categóricas'), 'variable-objetivo-vs-variables-categoricas')),
        html.Div([html.Div([html.Span(icon(i), className='drv-icon'), eyebrow(v.upper()), html.H3(c), html.Div([counter(round(r * 100, 1), 1, ' %')], className='drv-num'), html.Small(T('abandono') + ' · ' + s)], className='driver')
                  for i, v, c, r, s in drivers], className='drivers'),
        section(T('Interpretabilidad: SHAP y LIME'), src('MODELADO_AVANZADO', N('3.2 Interpretabilidad global y local: SHAP y LIME'), 'interpretabilidad-global-y-local-shap-y-lime')),
        card(field(T('Modelo explicado'), dropdown('explain-model', a.FINALISTS[:2], a.FINALISTS[0], {k: T(a.FINALIST_NAMES[k]) for k in a.FINALISTS[:2]})), html.Div(id='explain-output'), cls='controls-card'),
        grid(note('SHAP', T('Reparte cada predicción entre las variables usando valores de Shapley. Promediando su magnitud se obtiene la importancia global.')),
             note('LIME', T('Ajusta un modelo simple alrededor de un cliente para explicar su caso. Hay que revisar su fidelidad, no solo su estabilidad.'), 'violet')),
        note(T('Explicar no es causar'), T('Que el contrato mes a mes pese en el modelo no prueba que cambiar el contrato evite el abandono. Las atribuciones describen al modelo.'), 'amber')])


# ---------------------------------------------------------------- 7. conclusiones
def conclusiones(d):
    finds = [('layers', N('El perfil de cuenta delata el riesgo'), N('Contrato, antigüedad y servicio de internet concentran la señal, en el EDA y en SHAP.')),
             ('link', N('Empate en la cima'), N('Random Forest y XGBoost con SMOTE (F1 0.638) y la regresión logística (0.634) no se separan estadísticamente.')),
             ('scale', N('Balancear importa más que buscar'), N('SMOTE mejoró el F1 en 6 de 7 familias; cambiar de optimizador casi no movió el resultado.')),
             ('gauge', N('Probabilidades más honestas'), N('La calibración sigmoide mejoró Brier y ECE en 80 de 96 configuraciones.'))]
    limits = [N('Un único conjunto de datos: los folds no son bloques independientes.'), N('El test ya se había usado: es evidencia suplementaria.'),
              N('Los datos son una foto: no garantizan anticipar el abandono futuro.'), N('Sin costos de intervención, el F1 no se traduce en dinero.')]
    steps = [(N('Validación temporal'), N('Evaluar en clientes futuros.')), (N('Costos explícitos'), N('Elegir el umbral con el costo de cada error.')),
             (N('Modelo propuesto'), N('Superar a las referencias en desempeño y tiempo.'))]
    library_note = html.P(T('El texto de la biblioteca está disponible en español.'), className='note violet') if get_lang() == 'en' else None
    return html.Div([page_head('conclusiones', T('Qué aprendimos, qué sigue siendo incierto y qué falta para llevarlo a una campaña de retención.'), ['MODELADO_AVANZADO', 'paper']),
        section(T('Hallazgos principales'), src('MODELADO_AVANZADO', N('3.3 Conclusiones sustentadas por los resultados'), 'conclusiones-sustentadas-por-los-resultados')),
        html.Div([html.Div([html.Span(icon(i), className='drv-icon'), html.H3(T(t)), html.P(T(s))], className='finding') for i, t, s in finds], className='findings'),
        section(T('Limitaciones y trabajo futuro'), src('paper', N('4. Conclusiones'), 'conclusiones'), src('paper', N('5. Modelo propuesto: trabajo en curso'), 'modelo-propuesto-trabajo-en-curso')),
        grid(card(eyebrow(T('LIMITACIONES')), html.Ul([html.Li(T(x)) for x in limits])),
             card(eyebrow(T('PRÓXIMOS PASOS')), html.Div([html.Div([html.Span(f'0{k + 1}', className='obj-n'), html.Div([html.H4(T(t)), html.P(T(s))])], className='next') for k, (t, s) in enumerate(steps)]))),
        card(html.Div([html.Div([eyebrow(T('PAPER DEL PROYECTO')), html.H2(T('Modelo de clasificación binaria para la predicción de abandono de clientes')),
                                 html.P(T('Metodología completa, tablas, figuras y referencias en formato Springer Nature.'))]),
                       html.Div([html.A([icon('file'), T('Leer el paper')], href=BOOK + 'paper.html', target='_blank', className='btn primary'),
                                 html.A([icon('download'), 'PDF'], href=BOOK + 'paper.pdf', target='_blank', className='btn ghost')], className='cover-cta')], className='paper-card'), cls='glow'),
        section(T('Biblioteca del proyecto'), src('EDA_'), src('PREPROCESAMIENTO'), src('MODELADO_AVANZADO')),
        card(html.P(T('Todo el texto de los notebooks del libro, buscable por palabra clave.'), className='muted'), library_note,
             grid(field(T('Capítulo'), dropdown('reader-chapter', list(a.CHAPTERS), 'EDA_', {'EDA_': T('1 · Análisis exploratorio'), 'PREPROCESAMIENTO': T('2 · Preprocesamiento'), 'MODELADO_AVANZADO': T('3 · Modelado avanzado')})),
                  field(T('Buscar'), dcc.Input(id='reader-search', placeholder=T('Ej. SMOTE, calibración, contrato…'), type='search', debounce=True))), html.Div(id='reader-output'), cls='controls-card')])


# ---------------------------------------------------------------- 8. autoras
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


def chapters(chapter, pattern):
    chunks = [c for c in a.CHAPTERS[chapter] if re.search(pattern, c['title'], re.I)]
    return details([icon('book'), T('Leer el detalle del libro')], *[html.Details([html.Summary(c['title']), md(c['text'])], className='chapter') for c in chunks])


PAGES = {'inicio': lambda d: cover(), **{k: globals()[k] for k, _, _ in NAV}}

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
    footer = [html.Span(T('Señal perdida · Telco Customer Churn')), html.Span('Laura Rivera & Natalý Cárdenas · Universidad del Norte'), html.A(['Jupyter Book', icon('arrow', 'sm')], href=BOOK, target='_blank')]
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


@app.callback(Output('var-chip-output', 'children'), Input('var-chip', 'value'), State('lang', 'data'))
def var_chip(variable, lang='es'):
    return bi(variable, 'train', 'Todos', 'Todos', [0, 72], lang)


@app.callback(Output('uni-output', 'children'), Input('uni-var', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'), State('lang', 'data'))
def uni(variable, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    d = d.assign(Estado=outcome(d))
    if variable in a.NUMERIC:
        s = d[variable].dropna(); q = s.quantile([.25, .5, .75]); iqr = q.loc[.75] - q.loc[.25]; out = ((s < q.loc[.25] - 1.5 * iqr) | (s > q.loc[.75] + 1.5 * iqr)).sum()
        h = px.histogram(d, x=variable, color='Estado', color_discrete_map=outcome_colors(), nbins=35, barmode='stack', marginal='box', labels={variable: label(variable), 'Estado': ''})
        return html.Div([kpis(kpi(T('MEDIA'), round(s.mean(), 2), '', '', 2), kpi(T('MEDIANA'), round(s.median(), 2), '', '', 2), kpi(T('EXTREMOS IQR'), int(out)), kpi(T('VACÍOS'), int(d[variable].isna().sum()))),
                         chart(T('Distribución · ') + label(variable), h, T('{n} valores válidos · asimetría {s}. El color separa a quienes se van de quienes se quedan.').format(n=f'{len(s):,}', s=f'{s.skew():.2f}'), 420)])
    g = d.groupby([variable, 'Estado']).size().reset_index(name='n'); g[variable] = g[variable].map(cat).astype(str)
    f = px.bar(g, x=variable, y='n', color='Estado', color_discrete_map=outcome_colors(), text_auto=True, labels={'n': T('Clientes'), 'Estado': ''}); f.update_xaxes(title='')
    return chart(T('Frecuencias · ') + label(variable), f, T('{n} categorías en esta selección.').format(n=g[variable].nunique()), 400)


@app.callback(Output('bi-output', 'children'), Input('bi-var', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'), State('lang', 'data'))
def bi(variable, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    if variable in a.NUMERIC:
        d = d.assign(Estado=outcome(d))
        v = px.violin(d, x='Estado', y=variable, color='Estado', color_discrete_map=outcome_colors(), box=True, points=False, labels={variable: label(variable), 'Estado': ''})
        med = ', '.join(f'{k}: {m:.1f}' for k, m in d.groupby('Estado')[variable].median().items())
        return chart(T('¿Quién se va según ') + label(variable).lower() + '?', v, T('Medianas · ') + med, 380, False)
    g = a.category_rates(d, variable); g['c'] = g[variable].map(cat).astype(str); g = g.sort_values('tasa')
    f = px.bar(g, x='tasa', y='c', orientation='h', text=g.tasa.map('{:.1%}'.format), custom_data=['n', 'abandonos'], color='tasa', color_continuous_scale=RISK_SCALE, range_color=[0, .6])
    f.update_xaxes(tickformat='.0%', title=T('Tasa de abandono')); f.update_yaxes(title=''); f.update_coloraxes(showscale=False)
    f.update_traces(hovertemplate='%{y}<br>' + T('Abandono') + ': %{x:.1%}<br>n=%{customdata[0]} · ' + T('abandonos') + '=%{customdata[1]}<extra></extra>')
    top = g.iloc[-1]
    return chart(T('Tasa de abandono · ') + label(variable), f, T('Más frágil: {c} ({p}, n={n}). Asociación descriptiva, no causal.').format(c=top.c, p=f'{top.tasa:.1%}', n=f'{int(top.n):,}'),
                 max(260, 70 + 48 * len(g)), False)


@app.callback(Output('scatter-output', 'children'), Input('scatter-x', 'value'), Input('scatter-y', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'), State('lang', 'data'))
def scatter(x, y, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    sample = d.sample(min(2000, len(d)), random_state=42); sample = sample.assign(Estado=outcome(sample))
    f = px.scatter(sample, x=x, y=y, color='Estado', color_discrete_map=outcome_colors(), opacity=.55, labels={x: label(x), y: label(y), 'Estado': ''})
    rho = d[[x, y]].corr(method='spearman').iloc[0, 1]
    return html.Div([graph(f, 330), html.P(T('Muestra de {n} clientes · Spearman = {r}').format(n=f'{len(sample):,}', r=f'{rho:.3f}') if x != y else T('Elige dos variables distintas.'), className='insight')])


@app.callback(Output('models-output', 'children'), Input('model-family', 'value'), Input('model-balance', 'value'), Input('model-metric', 'value'), State('lang', 'data'))
def model_output(family, balance, metric, lang='es'):
    set_lang(lang); d = a.SUMMARY.copy()
    if family != 'Todas': d = d[d.model == family]
    if balance != 'Todos': d = d[d.balance == balance]
    d = d.sort_values(metric + '_mean', ascending=False); top = d.head(12).copy()
    conf = T('Configuración')
    top[conf] = top.model.map(model_name) + ' · ' + top.balance.map(balance_name) + ' · ' + top.optimizer.map(optimizer_name)
    f = px.bar(top.iloc[::-1], x=metric + '_mean', y=conf, error_x=metric + '_std', orientation='h', color='model', color_discrete_map=MODEL_COLORS, hover_data={metric + '_std': ':.3f', 'model': False})
    lo = max(0, top[metric + '_mean'].min() - .08)
    f.update_xaxes(title=T(METRICS[metric]) + ' · ' + T('media externa'), range=[lo, min(1, top[metric + '_mean'].max() + .05)]); f.update_yaxes(title='')
    heat = px.imshow(d.pivot_table(index='model', columns='balance', values=metric + '_mean', aggfunc='mean').rename(index=model_name, columns=balance_name), text_auto='.3f',
                     color_continuous_scale=VALUE_SCALE, aspect='auto', labels={'x': '', 'y': '', 'color': T(METRICS[metric])})
    view = d[['model', 'balance', 'optimizer', metric + '_mean', metric + '_std', 'recall_positive_mean', 'precision_positive_mean', 'roc_auc_mean']].copy()
    view = view.loc[:, ~view.columns.duplicated()]; view['model'] = view.model.map(model_name); view['balance'] = view.balance.map(balance_name); view['optimizer'] = view.optimizer.map(optimizer_name)
    return html.Div([grid(chart(T('Top {k} de {n} configuraciones').format(k=len(top), n=len(d)), f, T('Barras: media de 5 folds externos; bigotes: desviación estándar entre folds. El color identifica la familia.'), 480, False),
                          chart(T('Familia × balanceo'), heat, T('Promedio sobre optimizadores. Las columnas con SMOTE brillan más en casi todas las familias.'), 480, False)),
                     details([icon('table'), T('Tabla completa · {n} configuraciones').format(n=len(d))], table(view))])


@lru_cache(maxsize=1)
def anytime(): return pd.read_csv(a.DATA / 'anytime.csv')


OPT_COLORS = [PALETTE['teal'], PALETTE['coral'], PALETTE['violet'], PALETTE['amber']]


@app.callback(Output('opt-output', 'children'), Input('opt-family', 'value'), State('lang', 'data'))
def opt_output(family, lang='es'):
    set_lang(lang); d = anytime(); d = d[(d.model == family) & (d.replicate == 0)]
    b = T('Buscador'); step = T('Paso')
    g = d.groupby(['optimizer', 'evaluation']).best_inner_f1.mean().reset_index(); g[b] = g.optimizer.map(optimizer_name)
    frames = pd.concat([g[g.evaluation <= k].assign(**{step: k}) for k in range(1, int(g.evaluation.max()) + 1)])
    lo, hi = g.best_inner_f1.min(), g.best_inner_f1.max(); pad = (hi - lo) * .15 + .002
    f = px.line(frames, x='evaluation', y='best_inner_f1', color=b, animation_frame=step, markers=True, range_x=[0.5, g.evaluation.max() + .5], range_y=[lo - pad, hi + pad],
                labels={'evaluation': T('Evaluación'), 'best_inner_f1': T('Mejor F1 interno')}, color_discrete_sequence=OPT_COLORS)
    f.layout.updatemenus[0].buttons[0].args[1].update(frame={'duration': 220, 'redraw': False}, transition={'duration': 120})
    f.update(data=f.frames[-1].data); f.layout.sliders[0].active = len(f.frames) - 1   # arranca mostrando la carrera completa
    u = a.UNITS[(a.UNITS.model == family) & (a.UNITS.replicate == 0)].copy(); u[b] = u.optimizer.map(optimizer_name)
    bx = px.box(u, x=b, y='search_seconds', color=b, color_discrete_sequence=OPT_COLORS, labels={'search_seconds': T('Segundos de búsqueda')})
    return grid(chart(T('La carrera: mejor F1 acumulado'), f, T('Dale play para ver cómo cada buscador mejora evaluación tras evaluación.'), 430),
                chart(T('Tiempo de búsqueda'), bx, T('Segundos por unidad externa en el equipo del experimento.'), 430, False))


@app.callback(Output('cal-output', 'children'), Input('cal-family', 'value'), State('lang', 'data'))
def cal_output(family, lang='es'):
    set_lang(lang); d = a.UNITS[(a.UNITS.model == family) & (a.UNITS.replicate == 0)]
    g = d.groupby('balance')[['brier', 'cal_brier', 'ece', 'cal_ece']].mean().reset_index(); g.balance = g.balance.map(balance_name)
    if g.brier.isna().all(): return note(T('Sin probabilidad nativa'), T('SVM se configuró sin probabilidades nativas; solo la rama calibrada tiene Brier y ECE.'))
    r = T('Rama')
    m = g.melt(id_vars='balance', value_vars=['brier', 'cal_brier'], var_name=r, value_name='Brier'); m[r] = m[r].map({'brier': T('Sin calibrar'), 'cal_brier': T('Calibrado')})
    return html.Div([graph(px.bar(m, x='balance', y='Brier', color=r, barmode='group', color_discrete_sequence=[CHURN, LOW], labels={'balance': ''}), 300),
                     html.P(T('Menor es mejor. Promedio sobre folds externos y optimizadores.'), className='insight')])


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
    roc = go.Figure(go.Scatter(x=fpr, y=tpr, line_color=STAY, name='ROC', fill='tozeroy', fillcolor='rgba(27,167,196,.10)'))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line={'dash': 'dash', 'color': '#8792a8'}, name=T('Azar')))
    roc.add_trace(go.Scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode='markers', marker={'size': 14, 'color': CHURN, 'line': {'color': 'white', 'width': 2}}, name=T('Umbral actual')))
    roc.update_xaxes(title=T('Tasa de falsas alertas')); roc.update_yaxes(title=T('Tasa de detección'))
    cards = [kpi(k.upper() if k != 'Alertas' else T('ALERTAS'), round(v * 100, 1) if k != 'Alertas' else v, T('test histórico'), 'coral' if k == 'Recall' else '', 1 if k != 'Alertas' else 0, ' %' if k != 'Alertas' else '')
             for k, v in m.items()]
    return html.Div([kpis(*cards),
                     grid(card(html.H3(T('100 clientes del test')), html.Div(waffle, className='waffle'),
                               html.Div([html.Div([html.Span(className='w ' + k), f'{t}: ', html.B(f'{v:,}')], className='leg') for k, t, v in legend], className='legend')),
                          chart(T('Curva ROC y punto de operación'), roc, T('AUC = {a}. El punto coral se mueve con el umbral.').format(a=f'{roc_auc_score(p.y_true, p.score):.3f}'), 360))])


@app.callback(Output('explain-output', 'children'), Input('explain-model', 'value'), State('lang', 'data'))
def explain_output(experiment, lang='es'):
    set_lang(lang)
    path = a.DATA / 'interpretabilidad' / experiment
    g = pd.read_csv(path / 'importancia_global.csv').rename(columns={'Unnamed: 0': 'Variable'}).head(10)
    g['Nombre'] = g.Variable.map(label)
    fg = px.bar(g.iloc[::-1], x='mean_abs_shap', y='Nombre', orientation='h', color='mean_abs_shap', color_continuous_scale=[[0, PALETTE['violet']], [1, CHURN]], text=g.iloc[::-1].mean_abs_shap.map('{:.3f}'.format))
    fg.update_coloraxes(showscale=False); fg.update_yaxes(title=''); fg.update_xaxes(title=T('Media |SHAP|'))
    ids = [4267, 4386] if experiment == a.FINALISTS[0] else [2221, 4513]
    local = []
    for row in ids:
        s = pd.read_csv(path / f'shap_local_{row}.csv'); s = s.reindex(s.shap.abs().sort_values(ascending=False).index).head(8).sort_values('shap')
        eff = T('Efecto'); s[eff] = np.where(s.shap > 0, T('Empuja a irse'), T('Empuja a quedarse')); s['feature'] = s.feature.map(pretty)
        pred = a.predictions(experiment).set_index('row_id').loc[row]
        ok = pred.y_true == pred.y_pred
        f = px.bar(s, x='shap', y='feature', orientation='h', color=eff, color_discrete_map={T('Empuja a irse'): CHURN, T('Empuja a quedarse'): STAY}, labels={eff: ''}); f.update_yaxes(title=''); f.update_xaxes(title='SHAP')
        local.append(card(html.Div([html.H3(T('Cliente') + ' ' + a.RAW.customerID[row]), html.Span([icon('check' if ok else 'x', 'sm'), T('acierto') if ok else T('error')], className='pill ' + ('cyan' if ok else 'coral'))], className='card-head'),
                          html.P(T('Realidad: {r} · probabilidad del modelo: {p}').format(r=T('abandonó') if pred.y_true else T('permaneció'), p=f'{pred.probability:.1%}'), className='muted'), graph(f, 320),
                          details([icon('eye'), T('Ver explicación LIME')], html.Img(src=f'/evidence/{experiment}/lime_local_{row}.png', className='evidence', alt=f'LIME {row}'))))
    return html.Div([chart(T('Importancia global · ¿qué mira el modelo?'), fg, T('Promedio de |SHAP| en los 5,634 clientes de desarrollo. Mide cuánto pesa cada variable, no en qué dirección causal.'), 400, False),
                     html.H3(T('Dos clientes bajo la lupa'), className='section-title'), grid(*local)])


@app.callback(Output('explorer-table', 'children'), Input('explorer-cols', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'), State('lang', 'data'))
def explorer_table(cols, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    return table(d[cols or ['customerID']], id='data-table', page_size=12) if not d.empty else empty()


@app.callback(Output('download', 'data'), Input('export', 'n_clicks'), State('explorer-cols', 'value'), State('split', 'value'), State('contract', 'value'), State('internet', 'value'), State('tenure', 'value'), prevent_initial_call=True)
def export(n, cols, split, contract, internet, tenure):
    # Solo un clic real descarga: al volver a pintar la sección el botón nace con n_clicks = 0.
    if not n: return no_update
    return dcc.send_data_frame(a.cohort(split, contract, internet, tenure)[cols or ['customerID']].to_csv, 'telco_seleccion.csv', index=False)


@app.callback(Output('reader-output', 'children'), Input('reader-chapter', 'value'), Input('reader-search', 'value'), State('lang', 'data'))
def reader_output(chapter, search, lang='es'):
    set_lang(lang)
    search = (search or '').casefold(); chunks = [c for c in a.CHAPTERS[chapter] if search in c['text'].casefold() or search in c['title'].casefold()]
    return html.Div([html.P(T('{n} secciones encontradas').format(n=len(chunks)), className='muted'), *[html.Details([html.Summary(c['title']), md(c['text'])], className='chapter') for c in chunks]])


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
