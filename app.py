"""Señal Perdida: dashboard interactivo del proyecto Telco Customer Churn de Laura Rivera y Natalý Cárdenas."""
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

BOOK = 'https://nataly-cardenas.github.io/Telco_Churn_Entregable/'
REPO = 'https://github.com/Nataly-Cardenas/Telco_Churn_Entregable'
DASH_REPO = 'https://github.com/Nataly-Cardenas/Dash_Telco_Churn'
CYAN, VIOLET, CORAL, MINT, AMBER = '#22d3ee', '#8b5cf6', '#fb7185', '#34d399', '#fbbf24'
COLORS = {'Permanece': CYAN, 'Abandona': CORAL}
SEQ = [[0, '#0b1a33'], [.5, VIOLET], [1, CORAL]]
NAV = [('problema', 'El problema', '📡'), ('datos', 'Datos', '📊'), ('preparacion', 'Preparación', '🧪'), ('modelos', 'Arena de modelos', '🏁'),
       ('simulador', 'Simulador', '🎛️'), ('porque', '¿Por qué se van?', '🔍'), ('conclusiones', 'Conclusiones', '🧭'), ('autoras', 'Autoras', '👩‍💻')]
TITLES = {'inicio': 'Inicio', **{k: t for k, t, _ in NAV}}
DATA_PAGES = {'datos'}

app = Dash(__name__, suppress_callback_exceptions=True, title='Señal Perdida · Telco Customer Churn',
           external_stylesheets=['https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap'],
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
                      {'name': 'description', 'content': 'Dashboard interactivo para entender y anticipar el abandono de clientes de telecomunicaciones.'}])
server = app.server
app.index_string = app.index_string.replace('<html>', '<html lang="es">')


@server.route('/evidence/<path:filename>')
def evidence(filename):
    return send_from_directory(a.DATA / 'interpretabilidad', filename)


@server.route('/healthz')
def health(): return {'status': 'ok', 'clients': len(a.RAW), 'experiments': len(a.SUMMARY)}


# ---------------------------------------------------------------- componentes
def md(text): return dcc.Markdown(text, mathjax=True, className='prose')
def card(*children, cls=''): return html.Section(list(children), className='card ' + cls)
def note(title, text, cls=''): return html.Div([html.Strong(title), html.P(text)], className='note ' + cls)
def grid(*kids, cols=2): return html.Div(list(kids), className=f'grid g{cols}')
def options(values, labels=None): return [{'label': (labels or {}).get(v, a.ES.get(v, v) if isinstance(v, str) else v), 'value': v} for v in values]
def dropdown(id, values, value, labels=None, multi=False): return dcc.Dropdown(options(values, labels), value, id=id, clearable=False, multi=multi)
def field(text, control): return html.Div([html.Label(text), control], className='field')
def eyebrow(text): return html.Span(text, className='eyebrow')
def counter(value, decimals=0, suffix='', prefix=''):
    return html.Span(className='count', **{'data-count': str(value), 'data-decimals': str(decimals), 'data-suffix': suffix, 'data-prefix': prefix})
def kpi(label, value, sub='', accent='', decimals=0, suffix='', prefix=''):
    v = counter(value, decimals, suffix, prefix) if isinstance(value, (int, float)) else value
    return html.Div([html.Span(label, className='k-label'), html.Strong(v, className='k-value ' + accent), html.Small(sub)], className='kpi')
def kpis(*items): return html.Div(list(items), className='kpis')


def table(d, id=None, **kw):
    args = {'id': id} if id else {}
    return dash_table.DataTable(data=d.round(4).replace({np.nan: None}).to_dict('records'), columns=[{'name': c, 'id': c} for c in d.columns],
                                page_size=kw.pop('page_size', 10), sort_action='native', filter_action=kw.pop('filter_action', 'native'), style_table={'overflowX': 'auto'},
                                style_header={'backgroundColor': '#111c33', 'color': '#e2e8f0', 'border': 'none', 'fontWeight': '600'},
                                style_cell={'backgroundColor': '#0b1426', 'color': '#b6c2d9', 'border': 'none', 'borderBottom': '1px solid #1c2945', 'padding': '10px',
                                            'textAlign': 'left', 'fontFamily': 'Inter, sans-serif', 'fontSize': 12, 'maxWidth': 320, 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                                style_filter={'backgroundColor': '#0e1a31', 'color': '#e2e8f0'}, **args, **kw)


def style(f, height=340, legend=True, margin=None):
    f.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter, sans-serif', 'color': '#a8b5cc', 'size': 12}, colorway=[CYAN, CORAL, VIOLET, MINT, AMBER, '#60a5fa', '#f472b6'],
                    margin=margin or {'l': 20, 'r': 20, 't': 30, 'b': 30}, height=height, showlegend=legend,
                    legend={'orientation': 'h', 'y': 1.13, 'title_text': ''}, hoverlabel={'bgcolor': '#0b1426', 'bordercolor': CYAN, 'font_family': 'Inter'})
    f.update_xaxes(gridcolor='#16233d', zerolinecolor='#22324f'); f.update_yaxes(gridcolor='#16233d', zerolinecolor='#22324f')
    return f


def graph(f, height=340, legend=True, id=None, margin=None):
    args = {'id': id} if id else {}
    return dcc.Graph(figure=style(f, height, legend, margin), style={'height': height, 'width': '100%'}, config={'displaylogo': False, 'responsive': True,
                     'modeBarButtonsToRemove': ['lasso2d', 'select2d']}, **args)


def chart(title, f, insight='', height=340, legend=True):
    return card(html.Div([html.H3(title), html.Span('● en vivo', className='live')], className='card-head'), graph(f, height, legend),
                html.P(insight, className='insight') if insight else None)


def page_head(key, lead, chapter=None):
    link = html.A('Ver en el Jupyter Book ↗', href=BOOK + chapter + '.html', target='_blank', className='chip-link') if chapter else None
    return html.Div([html.Div([eyebrow('TELCO CUSTOMER CHURN · ' + TITLES[key].upper()), link], className='between'),
                     html.H1(TITLES[key]), html.P(lead, className='lead')], className='page-head')


def tabs(id, items):
    return dcc.Tabs([dcc.Tab(children=html.Div(content, className='tab-body'), label=label, value=str(i), className='tab', selected_className='tab on')
                     for i, (label, content) in enumerate(items)], value='0', id=id, className='tabs', parent_className='tabs-wrap')


def empty(): return note('Sin clientes en este segmento', 'Amplía los filtros para volver a calcular las visualizaciones.')
def scope(d): return html.Div([html.Span(className='live-dot'), f' {len(d):,} clientes en la selección · análisis descriptivo'], className='scope')


# ---------------------------------------------------------------- portada
@lru_cache(maxsize=1)
def cover_customers():
    """30 clientes reales del test con la probabilidad nativa del Random Forest finalista."""
    p = a.predictions(a.FINALISTS[0]).set_index('row_id')
    rng = np.random.default_rng(7)
    picks = list(rng.choice(p.index[p.probability >= .6], 12, replace=False)) + list(rng.choice(p.index[p.probability.between(.3, .6)], 8, replace=False)) \
        + list(rng.choice(p.index[p.probability < .3], 10, replace=False))
    rng.shuffle(picks)
    out = []
    for r in picks:
        c = a.RAW.loc[r]
        out.append({'id': c.customerID, 'contract': a.ES[c.Contract], 'internet': a.ES.get(c.InternetService, c.InternetService), 'payment': a.ES[c.PaymentMethod],
                    'tenure': int(c.tenure), 'monthly': float(c.MonthlyCharges), 'prob': round(float(p.loc[r, 'probability']), 3), 'churn': int(c.target), 'flags': flags(c)})
    return out


def flags(c):
    """Señales del perfil asociadas a más (▲) o menos (▼) abandono en el EDA."""
    out = []
    if c.Contract == 'Month-to-month': out.append('▲ Contrato mes a mes')
    if c.Contract == 'Two year': out.append('▼ Contrato a dos años')
    if c.InternetService == 'Fiber optic': out.append('▲ Fibra óptica')
    if c.PaymentMethod == 'Electronic check': out.append('▲ Paga con cheque electrónico')
    if c.tenure <= 12: out.append('▲ Menos de un año como cliente')
    if c.tenure > 48: out.append('▼ Más de 4 años como cliente')
    if c.TechSupport == 'No': out.append('▲ Sin soporte técnico')
    if c.OnlineSecurity == 'Yes': out.append('▼ Con seguridad en línea')
    return out[:4]


def phone():
    return html.Div(html.Div([
        html.Div([html.Span('9:41'), html.Span([html.B(), html.B(), html.B(), html.B()], className='sig'), html.Span('5G ▮', className='bat')], className='phone-status'),
        html.Div([
            html.Div([html.Span('TELCO · RADAR DE CLIENTES', className='pc-app'), html.Span('EN VIVO', className='pc-live')], className='between'),
            html.Div([html.Small('Cliente'), html.Strong(className='pc-id')], className='pc-row'),
            html.Div([html.Span(className='pc-contract pill'), html.Span(className='pc-internet pill'), html.Span(className='pc-payment pill')], className='pc-pills'),
            html.Div([html.Div([html.Small('Antigüedad'), html.Strong(className='pc-tenure')]), html.Div([html.Small('Cargo mensual'), html.Strong(className='pc-monthly')])], className='pc-grid'),
            html.Div([html.Small('Señales del perfil'), html.Ul(className='pc-flags')], className='pc-signals'),
            html.Div([html.Div(html.Div([html.Strong(className='ring-pct'), html.Small('prob. de abandono')], className='ring-label'), className='ring'),
                html.Div([html.Strong(className='pc-level'), html.Small('Random Forest + SMOTE · test histórico'), html.Small(className='pc-truth')], className='pc-verdict')], className='pc-risk'),
            html.Div(className='toast')], className='phone-body')], id='phone', className='phone', **{'data-customers': json.dumps(cover_customers())}),
        className='phone-stage')


def cover():
    return html.Div([
        html.Canvas(id='net-canvas', className='net-canvas'),
        html.Div(className='cover-glow'),
        html.Div([
            html.Div([
                eyebrow('TELCO CUSTOMER CHURN · UNIVERSIDAD DEL NORTE'),
                html.H1(['Señal', html.Br(), html.Span('perdida', className='glitch', **{'data-text': 'perdida'})], className='cover-title'),
                html.P(['Uno de cada cuatro clientes ', html.B('cuelga para siempre'), '. Descubre quiénes son, por qué se van y cómo anticiparlo con datos.'], className='cover-lead'),
                html.Div([
                    html.Div([counter(7043), html.Small('clientes analizados')]),
                    html.Div([counter(26.5, 1, ' %'), html.Small('abandonan el servicio')], className='hot'),
                    html.Div([counter(112), html.Small('modelos en competencia')]),
                    html.Div([counter(0.638, 3), html.Small('F1 del mejor modelo')])], className='cover-stats'),
                html.Div([dcc.Link('Entrar al dashboard →', href='#problema', className='btn primary'),
                          html.A('Jupyter Book ↗', href=BOOK, target='_blank', className='btn ghost')], className='cover-cta'),
                html.Div([html.Span('🖱️'), ' Los puntos rojos son clientes a punto de irse: ', html.B('pasa el cursor (o el dedo) sobre ellos para retenerlos.'),
                          html.Span([' Perdidos: ', html.B('0', id='cover-lost'), ' · Retenidos por ti: ', html.B('0', id='cover-saved')], className='score')], className='cover-hint'),
            ], className='cover-text'),
            phone()], className='cover-inner'),
        html.Div('Laura S. Rivera Figueredo · Natalý A. Cárdenas Izaquita', className='cover-foot')], className='cover')


# ---------------------------------------------------------------- 1. el problema
def people_row():
    leaving = {2, 7, 11, 16, 19}   # 5 de 20 ≈ 26.5 % de abandono
    return html.Div([html.Span(className='person' + (' leaving' if i in leaving else ''), style={'animationDelay': f'{(i % 5) * .9}s'}) for i in range(20)], className='people')


FLIP = [('🎯', 'Clasificación binaria', 'Estimar para cada cliente la probabilidad P(abandono | x) y separar a quienes se van (1) de quienes permanecen (0).'),
        ('⚖️', 'Desbalance de clases', 'Solo el 26.5 % abandona. SMOTE y ADASYN crean ejemplos sintéticos de la minoría para que el modelo no la ignore.'),
        ('🚧', 'Fuga de información', 'Imputar, codificar, escalar y balancear solo con datos de entrenamiento; si no, las métricas mienten.'),
        ('🌲', 'Siete modelos', 'Regresión logística (base), KNN, Naive Bayes, árbol, Random Forest, XGBoost y SVM compiten bajo las mismas reglas.'),
        ('🔁', 'Validación anidada', 'Un bucle interno elige hiperparámetros y uno externo mide el desempeño con datos que no participaron en la elección.'),
        ('📐', 'F1 y AUC', 'F1 = 2TP / (2TP + FP + FN) equilibra detectar abandonos y evitar falsas alertas; AUC mide qué tan bien ordena el riesgo.'),
        ('💡', 'SHAP y LIME', 'Explican qué variables empujan cada predicción. Describen al modelo, no causas del abandono.')]
OBJECTIVES = [('Describir', 'Caracterizar los datos, su calidad y el desbalance del abandono.'), ('Reservar', 'Separar 20 % del test antes de mirar los datos.'),
              ('Perfilar', 'Encontrar los segmentos con más cancelaciones.'), ('Preparar', 'Transformar sin fuga de información.'),
              ('Referenciar', 'Entrenar una regresión logística como modelo base.'), ('Comparar', '112 configuraciones con validación cruzada anidada.'),
              ('Explicar', 'Identificar con SHAP y LIME qué variables influyen.'), ('Compartir', 'Publicar libro, paper y este dashboard.')]
GROUPS = {'💳 Cuenta': ['Contract', 'PaymentMethod', 'PaperlessBilling', 'tenure', 'MonthlyCharges', 'TotalCharges'],
          '📶 Servicios': ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'],
          '👥 Demografía': ['gender', 'SeniorCitizen', 'Partner', 'Dependents']}


def problema(d):
    return html.Div([
        page_head('problema', 'Perder un cliente cuesta más que conservarlo. Este proyecto busca anticipar quién está por cancelar su servicio para actuar a tiempo.', 'intro'),
        html.Div([html.Div([eyebrow('EL RETO'), html.H2(['1 de cada 4 clientes ', html.Span('se desconecta', className='txt-coral')]),
                            html.P('El conjunto Telco Customer Churn de IBM reúne servicios contratados, datos de cuenta y perfil demográfico de cada cliente, '
                                   'junto con si canceló en el último mes. Dos dificultades: los que se van son minoría, y los datos son una foto de un momento.'),
                            people_row()], className='hero-text'),
                  html.Div([kpi('CLIENTES', 7043, 'una fila por cliente'), kpi('ABANDONOS', 1869, 'cancelaron en el último mes', 'coral'),
                            kpi('TASA DE ABANDONO', 26.5, 'la clase minoritaria', 'coral', 1, ' %'), kpi('VARIABLES', 19, 'predictoras candidatas')], className='kpis k2')], className='hero'),
        html.H2('¿Qué buscamos?', className='section-title'),
        card(eyebrow('OBJETIVO GENERAL'), html.P('Identificar a los clientes con mayor riesgo de abandonar el servicio y los factores asociados, comparando de forma reproducible '
                                              'modelos de clasificación que detecten la mayor cantidad de clientes en riesgo sin generar un exceso de falsas alertas.', className='big-text'), cls='glow'),
        html.Div([html.Div([html.Span(f'{i + 1:02d}', className='obj-n'), html.H4(t), html.P(s)], className='obj') for i, (t, s) in enumerate(OBJECTIVES)], className='objectives'),
        html.H2('Marco teórico en 7 tarjetas', className='section-title'), html.P('Pasa el cursor (o toca) cada tarjeta para voltearla.', className='muted'),
        html.Div([html.Div(html.Div([html.Div([html.Span(i, className='flip-icon'), html.H4(t)], className='flip-front'), html.Div([html.H4(t), html.P(s)], className='flip-back')],
                                    className='flip-inner'), className='flip', tabIndex=0) for i, t, s in FLIP], className='flips'),
        html.H2('Las variables del problema', className='section-title'),
        card(html.P('Elige una variable para ver cómo cambia la tasa de abandono entre sus categorías (conjunto de entrenamiento).', className='muted'),
             html.Div([html.Span(g, className='chip-group') for g in GROUPS], className='chip-groups'),
             dcc.RadioItems([{'label': html.Span(a.LABELS[v], className='grp-' + str(i)), 'value': v} for i, vs in enumerate(GROUPS.values()) for v in vs], 'Contract', id='var-chip', className='chips', inline=True),
             html.Div(id='var-chip-output'))])


# ---------------------------------------------------------------- 2. datos
def datos(d):
    if d.empty: return html.Div([page_head('datos', 'Explora el conjunto de datos con los filtros superiores.', 'EDA_'), empty()])
    missing = d[a.NUMERIC + a.CATEGORICAL].isna().sum().rename_axis('Variable').reset_index(name='Vacíos')
    audit = pd.DataFrame({'Control': ['Clientes', 'Identificadores repetidos', 'Filas repetidas', 'TotalCharges vacío', 'Vacío con tenure ≠ 0'],
                          'Resultado': [len(d), int(d.customerID.duplicated().sum()), int(d[a.RAW.columns[:21]].duplicated().sum()), int(d.TotalCharges.isna().sum()),
                                        int((d.TotalCharges.isna() & (d.tenure != 0)).sum())]})
    split = go.Figure(go.Pie(labels=['Entrenamiento', 'Test histórico'], values=[5634, 1409], hole=.72, marker_colors=[CYAN, VIOLET], textinfo='percent', sort=False))
    split.add_annotation(text='<b>80 / 20</b><br>estratificado', showarrow=False, font_size=16)
    calidad = [kpis(kpi('FILAS', 7043, 'una por cliente'), kpi('ENTRENAMIENTO', 5634, '80 % · aquí se hace el EDA'), kpi('TEST', 1409, '20 % · reservado'),
                    kpi('VACÍOS', 11, 'TotalCharges con tenure = 0', 'amber')),
               grid(chart('Partición antes de mirar los datos', split, 'El test se separa primero para que ninguna decisión se tome con él.', 300, False),
                    card(html.H3('Auditoría de la selección'), table(audit, filter_action='none'), html.P('Los 11 vacíos son clientes nuevos sin facturación acumulada: '
                         'se reemplazan por 0 en el preprocesamiento, no por un promedio ficticio.', className='insight'))),
               html.Details([html.Summary('📖 Diccionario de variables'), table(pd.DataFrame([{'Variable': c, 'Significado': a.LABELS.get(c, 'Identificador' if c == 'customerID' else 'Abandono'),
                             'Tipo': 'Numérica' if c in a.NUMERIC else 'Categórica', 'Rol': 'Objetivo' if c == 'Churn' else 'No predictor' if c == 'customerID' else 'Predictor'}
                             for c in a.RAW.columns[:21]]), page_size=21)], className='details')]
    distribuciones = [card(field('Variable', dropdown('uni-var', a.NUMERIC + a.CATEGORICAL, 'tenure', a.LABELS))), html.Div(id='uni-output')]
    sun = d.assign(Contrato=d.Contract.map(a.ES), Internet=d.InternetService.map(a.ES), Pago=d.PaymentMethod.map(a.ES)).groupby(['Contrato', 'Internet', 'Pago']).target.agg(['size', 'mean']).reset_index()
    sb = px.sunburst(sun, path=['Contrato', 'Internet', 'Pago'], values='size', color='mean', color_continuous_scale=SEQ, range_color=[0, .7], labels={'mean': 'Abandono', 'size': 'Clientes'})
    sb.update_traces(hovertemplate='<b>%{label}</b><br>Clientes: %{value}<br>Abandono: %{color:.1%}<extra></extra>', insidetextorientation='radial')
    segmentos = [grid(chart('Mapa de riesgo · haz clic para profundizar', sb, 'Contrato → internet → método de pago. El color es la tasa de abandono: el rojo marca '
                            'los perfiles más frágiles, como mes a mes + fibra óptica + cheque electrónico.', 470, False), tenure_race(d)),
                 card(field('Comparar abandono por', dropdown('bi-var', a.CATEGORICAL + a.NUMERIC, 'PaymentMethod', a.LABELS))), html.Div(id='bi-output')]
    corr = d[a.NUMERIC + ['target']].rename(columns={**a.LABELS, 'target': 'Abandono'}).corr(method='spearman')
    relaciones = [grid(chart('Correlaciones de Spearman', px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale=[[0, CYAN], [.5, '#0b1426'], [1, CORAL]]),
                             'Antigüedad y cargo acumulado comparten mucha información (ρ ≈ 0.89): por eso se crea el cargo promedio mensual en lugar de borrar una.', 380, False),
                       card(html.H3('Dos variables a la vez'), grid(field('Eje X', dropdown('scatter-x', a.NUMERIC, 'tenure', a.LABELS)), field('Eje Y', dropdown('scatter-y', a.NUMERIC, 'MonthlyCharges', a.LABELS))),
                            html.Div(id='scatter-output')))]
    explorador = [card(field('Columnas visibles', dropdown('explorer-cols', list(a.RAW.columns[:21]), ['customerID', 'Contract', 'InternetService', 'tenure', 'MonthlyCharges', 'Churn'], multi=True)),
                       html.Div(id='explorer-table'), html.Button('Descargar selección CSV ↓', id='export', className='btn primary'),
                       html.P('El CSV incluye toda la selección de los filtros superiores con las columnas elegidas.', className='insight'))]
    return html.Div([page_head('datos', 'Del archivo original a los patrones de abandono. Los filtros superiores recalculan todo lo que ves aquí.', 'EDA_'), scope(d),
                     kpis(kpi('CLIENTES', len(d), 'en la selección'), kpi('ABANDONO', round(d.target.mean() * 100, 2), f'{int(d.target.sum()):,} clientes', 'coral', 2, ' %'),
                          kpi('ANTIGÜEDAD MEDIA', round(d.tenure.mean(), 1), 'meses', '', 1), kpi('CARGO MENSUAL', round(d.MonthlyCharges.mean(), 2), 'promedio', '', 2, '', '$')),
                     tabs('datos-tabs', [('🧹 Calidad', calidad), ('📈 Distribuciones', distribuciones), ('🔥 Segmentos de riesgo', segmentos), ('🕸️ Relaciones', relaciones), ('🗂️ Explorador', explorador)])])


def tenure_race(d):
    """Animación: la tasa de abandono por contrato a medida que crece la antigüedad."""
    bins = [-1, 6, 12, 24, 36, 48, 60, 72]; names = ['0–6', '7–12', '13–24', '25–36', '37–48', '49–60', '61–72']
    g = d.assign(Tramo=pd.cut(d.tenure, bins, labels=names), Contrato=d.Contract.map(a.ES)).groupby(['Tramo', 'Contrato'], observed=False).target.agg(tasa='mean', n='size').reset_index().fillna({'tasa': 0})
    g['txt'] = g.tasa.map('{:.0%}'.format)
    f = px.bar(g, x='Contrato', y='tasa', color='Contrato', animation_frame='Tramo', range_y=[0, .75], text='txt',
               color_discrete_map={'Mes a mes': CORAL, 'Un año': VIOLET, 'Dos años': CYAN}, custom_data=['n'], category_orders={'Contrato': ['Mes a mes', 'Un año', 'Dos años']})
    f.update_traces(hovertemplate='%{x}<br>Abandono: %{y:.1%}<br>n=%{customdata[0]}<extra></extra>', textposition='outside')
    f.update_yaxes(tickformat='.0%', title='Tasa de abandono'); f.update_xaxes(title='')
    f.layout.updatemenus[0].buttons[0].args[1].update(frame={'duration': 900, 'redraw': True}, transition={'duration': 500})
    for fr, nm in zip(f.frames, names): fr.layout = go.Layout(title_text=f'Antigüedad {nm} meses')
    return chart('▶ Dale play: el riesgo cae con la antigüedad', f, 'Cada fotograma es un tramo de antigüedad. Los clientes nuevos con contrato mes a mes son los más '
                 'propensos a irse; con los años la tasa se desploma.', 470, False)


# ---------------------------------------------------------------- 3. preparación
STEPS = [('🎯', 'Objetivo', 'Churn Yes/No → 1/0. customerID solo para control.'), ('🧽', 'Limpieza', 'TotalCharges a número; vacíos con tenure 0 → 0.'),
         ('🧩', 'Categorías', '“Sin internet / sin teléfono” → “No”.'), ('➗', 'Nueva variable', 'AvgChargePerMonth = TotalCharges / tenure.'),
         ('🔢', 'Codificar y escalar', 'One-Hot + StandardScaler ajustados en cada fold.'), ('⚖️', 'Balancear', 'Ninguno, pesos, SMOTE o ADASYN; solo en train.'),
         ('🤖', 'Modelo', 'Se entrena y se evalúa en el fold reservado.')]


def preparacion(d):
    abl = pd.DataFrame({'Variable retirada': ['gender', 'PhoneService', 'Ambas', 'PhoneService + MultipleLines', 'AvgChargePerMonth'], 'ΔAUC': [.0002, .0002, .0005, -.0006, .0006]})
    fa = px.bar(abl, x='ΔAUC', y='Variable retirada', orientation='h', color='ΔAUC', color_continuous_scale=[[0, CORAL], [.5, '#334155'], [1, MINT]], range_color=[-.001, .001])
    base = pd.DataFrame({'Métrica': ['AUC', 'Recall', 'Precision', 'Accuracy'], 'Valor': [.8419, .7861, .5052, .7388]})
    fb = px.bar(base, x='Valor', y='Métrica', orientation='h', text=base.Valor.map('{:.3f}'.format), color='Métrica', color_discrete_sequence=[CYAN, CORAL, VIOLET, MINT])
    fb.update_xaxes(range=[0, 1], title=''); fb.update_yaxes(title='')
    fa.update_coloraxes(showscale=False); fa.update_xaxes(tickformat='.4f', title='Cambio en AUC al retirar'); fa.update_yaxes(title='')
    return html.Div([page_head('preparacion', 'Siete estaciones por las que pasa cada cliente antes de llegar al modelo, todas ajustadas sin espiar el test.', 'PREPROCESAMIENTO'),
        card(html.H3('El pipeline en movimiento'), html.Div([html.Div(className='flow-line'), *[html.Div([html.Span(i, className='st-icon'), html.H4(t), html.P(s)], className='station',
             style={'animationDelay': f'{k * .15}s'}) for k, (i, t, s) in enumerate(STEPS)]], className='pipeline')),
        kpis(kpi('AUC IMPUTANDO CON 0', .8459, 'CV en entrenamiento', '', 4), kpi('AUC CON MEDIANA', .8460, 'diferencia despreciable', '', 4),
             kpi('ETIQUETAS PERMUTADAS', .49, 'control: azar', 'coral', 2), kpi('COLUMNAS FINALES', 24, 'tras codificar')),
        grid(chart('Modelo base: regresión logística en test', fb, 'Detecta casi 8 de cada 10 abandonos, pero la mitad de sus alertas son falsas: hay margen de mejora.', 300, False),
             chart('Ablación: ¿qué pasa si quito una variable?', fa, 'Retirar gender o PhoneService casi no cambia el AUC; las diferencias están dentro del ruido entre folds.', 300, False)),
        grid(note('¿Por qué SMOTE solo en entrenamiento?', 'Si se balanceara antes de dividir, ejemplos sintéticos derivados de clientes de validación contaminarían el entrenamiento.'),
             note('Limitación honesta', 'SMOTE y ADASYN se aplicaron después del One-Hot y pueden producir indicadores fraccionarios; se reporta tal cual.', 'amber')),
        chapters('PREPROCESAMIENTO', r'4\.|7\.4|8\.')])


# ---------------------------------------------------------------- 4. modelos
def modelos(d):
    s = a.SUMMARY.assign(Modelo=a.SUMMARY.model.map(a.MODELS), Balanceo=a.SUMMARY.balance.map(a.BALANCES), Optimizador=a.SUMMARY.optimizer.map(a.OPTIMIZERS))
    f3 = px.scatter_3d(s, x='recall_positive_mean', y='precision_positive_mean', z='roc_auc_mean', color='Modelo', size='f1_positive_mean', size_max=16, opacity=.85,
                       hover_data={'Balanceo': True, 'Optimizador': True, 'f1_positive_mean': ':.3f'}, labels={'recall_positive_mean': 'Recall', 'precision_positive_mean': 'Precision', 'roc_auc_mean': 'AUC', 'f1_positive_mean': 'F1'})
    axis = {'gridcolor': '#1c2945', 'showbackground': False, 'zerolinecolor': '#22324f'}
    f3.update_layout(scene={'bgcolor': 'rgba(0,0,0,0)', 'xaxis': axis, 'yaxis': axis, 'zaxis': axis, 'camera': {'eye': {'x': 1.6, 'y': 1.5, 'z': .9}}})
    ranking = [card(html.Div([field('Familia', dropdown('model-family', ['Todas'] + list(a.MODELS), 'Todas', a.MODELS)), field('Balanceo', dropdown('model-balance', ['Todos'] + list(a.BALANCES), 'Todos', a.BALANCES)),
                              field('Métrica', dropdown('model-metric', ['f1_positive', 'recall_positive', 'precision_positive', 'roc_auc', 'accuracy'], 'f1_positive', METRICS))], className='controls')),
               html.Div(id='models-output'),
               chart('Galaxia de 112 configuraciones · gira con el mouse', f3, 'Cada punto es una configuración; el tamaño es su F1. Las más grandes y altas combinan buen recall, precisión y AUC.', 560)]
    carrera = [card(field('Familia', dropdown('opt-family', list(a.MODELS), 'random_forest', a.MODELS))), html.Div(id='opt-output'),
               note('Sin ganador universal', 'Con 24 evaluaciones los cuatro buscadores terminan prácticamente empatados; las diferencias solo aparecen con presupuestos bajos.')]
    calib = [grid(card(html.H3('Probabilidades que dicen la verdad'), md('El balanceo distorsiona las probabilidades. La **calibración sigmoide** (Platt) se ajustó dentro del '
                       'entrenamiento y redujo **Brier** y **ECE** a la vez en **80 de 96** configuraciones.\n\n- **Brier**: error cuadrático de la probabilidad (menor es mejor).\n'
                       '- **ECE**: distancia entre la confianza y la frecuencia real.')), card(field('Familia', dropdown('cal-family', list(a.MODELS), 'random_forest', a.MODELS)), html.Div(id='cal-output')))]
    return html.Div([page_head('modelos', 'Siete familias × cuatro balanceos × cuatro buscadores = 112 configuraciones compitiendo bajo las mismas reglas.', 'MODELADO_AVANZADO'),
                     html.Div([podium(), html.Div([kpi('CONFIGURACIONES', 112, '7 × 4 × 4'), kpi('UNIDADES EXTERNAS', 1680, '112 × 3 semillas × 5 folds'),
                                                   kpi('FOLDS', '5 × 3', 'externos × internos'), kpi('PRESUPUESTO', 24, 'evaluaciones por búsqueda')], className='kpis k2')], className='hero'),
                     tabs('model-tabs', [('🏆 Ranking', ranking), ('🏎️ Carrera de buscadores', carrera), ('🎚️ Calibración', calib), ('📏 Evidencia estadística', estadistica())])])


METRICS = {'f1_positive': 'F1 positivo', 'recall_positive': 'Recall', 'precision_positive': 'Precision', 'roc_auc': 'ROC-AUC', 'accuracy': 'Accuracy'}


def podium():
    p = [('🥈', 'XGBoost', 'SMOTE · Random', .638, 'second'), ('🥇', 'Random Forest', 'SMOTE · TPE', .638, 'first'), ('🥉', 'Reg. logística', 'SMOTE · Grid', .634, 'third')]
    return card(eyebrow('PODIO · F1 MEDIO EXTERNO'), html.Div([html.Div([html.Span(m, className='medal'), html.Strong(n), html.Small(c), html.Div(counter(v, 3), className='bar-podium')],
                className='pod ' + k) for m, n, c, v, k in p], className='podium'), html.P('Empate técnico: la diferencia entre RF y XGBoost es 0.0004, menor que la variación entre folds.', className='insight'))


def estadistica():
    st = a.DATA / 'estadistica'; f = json.loads((st / 'friedman.json').read_text(encoding='utf-8'))
    test = pd.DataFrame({'Modelo': ['Random Forest', 'XGBoost', 'Reg. logística'], 'Accuracy': [.768, .769, .747], 'Precision': [.547, .546, .515], 'Recall': [.735, .765, .794],
                         'F1': [.627, .637, .625], 'AUC': [.844, .842, .842]})
    fr = px.line_polar(test.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'), r='Valor', theta='Métrica', color='Modelo', line_close=True, range_r=[.45, .85],
                       color_discrete_sequence=[CYAN, CORAL, VIOLET])
    fr.update_traces(fill='toself', opacity=.55); fr.update_layout(polar={'bgcolor': 'rgba(0,0,0,0)', 'radialaxis': {'gridcolor': '#1c2945'}, 'angularaxis': {'gridcolor': '#1c2945'}})
    return [kpis(kpi('FRIEDMAN χ²', round(f['statistic'], 2), f'Kendall W = {f["kendall_W"]:.2f}', '', 2), kpi('PARES NEMENYI < 0.05', 31, 'de 6,216 · exploratorio'),
                 kpi('p DELONG (HOLM)', 1.0, 'los tres pares', '', 1), kpi('ΔAUC RF − XGB', .0017, 'IC [−0.0034, 0.0070]', '', 4)),
            grid(chart('Finalistas en el test histórico', fr, 'Los tres finalistas se superponen: la regresión logística detecta más (Recall) y los ensambles aciertan más sus alertas (Precision).', 400),
                 card(html.H3('¿Hay un ganador?'), md('**No de forma demostrable.** Friedman detecta diferencias entre las 112 configuraciones, pero Nemenyi no separa a las mejores. '
                      'DeLong con corrección de Holm da *p* = 1.0 en los tres pares y todos los intervalos bootstrap del AUC incluyen el cero.\n\n'
                      'No detectar diferencias **no** prueba que sean equivalentes, y el test ya había sido expuesto: es evidencia suplementaria.'))),
            card(html.H3('Comparaciones DeLong'), table(pd.read_csv(st / 'delong_holm.csv'), filter_action='none'))]


# ---------------------------------------------------------------- 5. simulador
PAY = list(a.RAW.PaymentMethod.unique())


def simulador(d):
    perfil = [grid(card(html.H3('Arma un cliente'), field('Contrato', dropdown('sim-contract', ['Month-to-month', 'One year', 'Two year'], 'Month-to-month')),
                        field('Internet', dropdown('sim-internet', ['Fiber optic', 'DSL', 'No'], 'Fiber optic')), field('Método de pago', dropdown('sim-payment', PAY, 'Electronic check')),
                        field('Antigüedad (meses)', dcc.Slider(0, 72, 1, value=4, marks={0: '0', 24: '24', 48: '48', 72: '72'}, id='sim-tenure', tooltip={'placement': 'bottom'})),
                        field('Cargo mensual ($)', dcc.Slider(18, 119, 1, value=85, marks={20: '20', 60: '60', 100: '100'}, id='sim-monthly', tooltip={'placement': 'bottom'}))),
                   html.Div(id='sim-output')),
              note('¿De dónde sale este riesgo?', 'Es la tasa de abandono observada en clientes de entrenamiento con un perfil parecido, no una predicción del modelo. '
                   'Sirve para explorar patrones; para la predicción del modelo usa la pestaña de clientes reales.')]
    ids = a.TEST_IDS
    reales = [card(html.Div([field('Cliente del test histórico', dcc.Dropdown([{'label': a.RAW.customerID[i], 'value': i} for i in ids], int(ids[0]), id='real-id', clearable=False)),
                             html.Button('🎲 Cliente al azar', id='real-random', className='btn ghost', n_clicks=0)], className='controls'),
                    html.Div(id='real-output'))]
    umbral = [card(html.Div([field('Modelo finalista', dropdown('threshold-model', a.FINALISTS, a.FINALISTS[0], a.FINALIST_NAMES)),
                             field('Umbral de alerta', dcc.Slider(.05, .95, .01, value=.5, marks={.1: '0.10', .3: '0.30', .5: '0.50', .7: '0.70', .9: '0.90'}, id='threshold', tooltip={'always_visible': True}))],
                            className='controls'), html.Div(id='threshold-output')),
              note('Decisión de negocio', 'Bajar el umbral detecta más clientes en riesgo pero dispara más falsas alertas. Elegirlo bien requiere conocer el costo de cada contacto y de cada cliente perdido.')]
    return html.Div([page_head('simulador', 'Juega con perfiles, revisa clientes reales y decide dónde poner la alarma de retención.'),
                     tabs('sim-tabs', [('🧑‍💼 Perfil de riesgo', perfil), ('📱 Clientes reales', reales), ('🚨 Umbral de alerta', umbral)])])


def gauge(value, title, color):
    f = go.Figure(go.Indicator(mode='gauge+number', value=value * 100, number={'suffix': ' %', 'font': {'size': 34}}, title={'text': title, 'font': {'size': 13}},
                               gauge={'axis': {'range': [0, 100], 'tickcolor': '#475569'}, 'bar': {'color': color, 'thickness': .28}, 'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                                      'steps': [{'range': [0, 35], 'color': 'rgba(52,211,153,.12)'}, {'range': [35, 60], 'color': 'rgba(251,191,36,.12)'}, {'range': [60, 100], 'color': 'rgba(251,113,133,.15)'}]}))
    return f


GAUGE_M = {'l': 34, 'r': 34, 't': 50, 'b': 10}


def level(p): return ('alto', CORAL) if p >= .6 else ('medio', AMBER) if p >= .35 else ('bajo', MINT)


def signal(p):
    lit = max(1, round((1 - p) * 5))
    return html.Div([html.B(className='lit' if i < lit else '', style={'height': f'{12 + i * 9}px'}) for i in range(5)], className='signal ' + level(p)[0])


# ---------------------------------------------------------------- 6. por qué se van
def porque(d):
    t = a.RAW.loc[a.TRAIN_IDS]
    drivers = [('📝', 'Contrato', 'Mes a mes', t[t.Contract == 'Month-to-month'].target.mean(), 'vs 2.9 % con dos años'),
               ('🌐', 'Internet', 'Fibra óptica', t[t.InternetService == 'Fiber optic'].target.mean(), 'vs 7.3 % sin internet'),
               ('⏳', 'Antigüedad', 'Primer año', t[t.tenure <= 12].target.mean(), f'vs {t[t.tenure > 48].target.mean():.1%} después de 4 años')]
    return html.Div([page_head('porque', 'Lo que los modelos aprendieron: las variables que empujan el riesgo, y cómo explicaron aciertos y errores concretos.', 'MODELADO_AVANZADO'),
        html.Div([html.Div([html.Span(i, className='drv-icon'), eyebrow(v.upper()), html.H3(c), html.Div([counter(round(r * 100, 1), 1, ' %')], className='drv-num'), html.Small('abandono · ' + s)], className='driver')
                  for i, v, c, r, s in drivers], className='drivers'),
        card(field('Modelo explicado', dropdown('explain-model', a.FINALISTS[:2], a.FINALISTS[0], a.FINALIST_NAMES)), html.Div(id='explain-output')),
        grid(note('SHAP', 'Reparte cada predicción entre las variables usando valores de Shapley. Promediando su magnitud se obtiene la importancia global.'),
             note('LIME', 'Ajusta un modelo simple alrededor de un cliente para explicar su caso. Hay que revisar su fidelidad, no solo su estabilidad.', 'violet')),
        note('Explicar no es causar', 'Que el contrato mes a mes pese en el modelo no prueba que cambiar el contrato evite el abandono. Las atribuciones describen al modelo.', 'amber')])


# ---------------------------------------------------------------- 7. conclusiones
def conclusiones(d):
    finds = [('🧩', 'El perfil de cuenta delata el riesgo', 'Contrato, antigüedad y servicio de internet concentran la señal, en el EDA y en SHAP.'),
             ('🤝', 'Empate en la cima', 'Random Forest y XGBoost con SMOTE (F1 0.638) y la regresión logística (0.634) no se separan estadísticamente.'),
             ('⚖️', 'Balancear importa más que buscar', 'SMOTE mejoró el F1 en 6 de 7 familias; cambiar de optimizador casi no movió el resultado.'),
             ('🎯', 'Probabilidades más honestas', 'La calibración sigmoide mejoró Brier y ECE en 80 de 96 configuraciones.')]
    return html.Div([page_head('conclusiones', 'Qué aprendimos, qué sigue siendo incierto y qué falta para llevarlo a una campaña de retención.', 'paper'),
        html.Div([html.Div([html.Span(i, className='drv-icon'), html.H3(t), html.P(s)], className='finding') for i, t, s in finds], className='findings'),
        grid(card(eyebrow('LIMITACIONES'), html.Ul([html.Li(x) for x in ['Un único conjunto de datos: los folds no son bloques independientes.', 'El test ya se había usado: es evidencia suplementaria.',
                                                                         'Los datos son una foto: no garantizan anticipar el abandono futuro.', 'Sin costos de intervención, el F1 no se traduce en dinero.']])),
             card(eyebrow('PRÓXIMOS PASOS'), html.Div([html.Div([html.Span(f'0{k + 1}', className='obj-n'), html.Div([html.H4(t), html.P(s)])], className='next') for k, (t, s) in enumerate(
                 [('Validación temporal', 'Evaluar en clientes futuros.'), ('Costos explícitos', 'Elegir el umbral con el costo de cada error.'), ('Modelo propuesto', 'Superar a las referencias en desempeño y tiempo.')])]))),
        card(html.Div([html.Div([eyebrow('PAPER DEL PROYECTO'), html.H2('Modelo de clasificación binaria para la predicción de abandono de clientes'),
                                 html.P('Metodología completa, tablas, figuras y referencias en formato Springer Nature.')]),
                       html.Div([html.A('📄 Leer el paper ↗', href=BOOK + 'paper.html', target='_blank', className='btn primary'), html.A('⬇ PDF', href=BOOK + 'paper.pdf', target='_blank', className='btn ghost')], className='cover-cta')],
                      className='paper-card'), cls='glow'),
        html.H2('Biblioteca del proyecto', className='section-title'),
        card(html.P('Todo el texto de los notebooks del libro, buscable por palabra clave.', className='muted'),
             grid(field('Capítulo', dropdown('reader-chapter', list(a.CHAPTERS), 'EDA_', {'EDA_': '1 · Análisis exploratorio', 'PREPROCESAMIENTO': '2 · Preprocesamiento', 'MODELADO_AVANZADO': '3 · Modelado avanzado'})),
                  field('Buscar', dcc.Input(id='reader-search', placeholder='Ej. SMOTE, calibración, contrato…', type='search', debounce=True))), html.Div(id='reader-output'))])


# ---------------------------------------------------------------- 8. autoras
def autoras(d):
    people = [('LR', 'Laura S. Rivera Figueredo', 'sriveral@uninorte.edu.co', 'https://github.com/lauraformore'),
              ('NC', 'Natalý A. Cárdenas Izaquita', 'nizaquita@uninorte.edu.co', 'https://github.com/Nataly-Cardenas')]
    return html.Div([page_head('autoras', 'El equipo detrás del proyecto y los recursos para profundizar.'),
        html.Div([html.Div([html.Div(html.Span(i), className='avatar'), html.H2(n), html.P('Ciencia de Datos · Universidad del Norte', className='muted'),
                            html.Div([html.A('✉ ' + m, href='mailto:' + m, className='chip-link'), html.A('GitHub ↗', href=g, target='_blank', className='chip-link')], className='author-links')],
                           className='author') for i, n, m, g in people], className='authors'),
        html.P('Departamento de Matemáticas, Física y Ciencia de Datos · Universidad del Norte · Barranquilla, Colombia', className='center muted'),
        html.A([html.Span('📘', className='book-icon'), html.Div([eyebrow('JUPYTER BOOK'), html.H2('Lee el proyecto completo'), html.P('Introducción, marco teórico, EDA, preprocesamiento, modelado y paper.')]),
                html.Span('↗', className='book-arrow')], href=BOOK, target='_blank', className='book-cta'),
        grid(html.A([html.H4('💻 Código y notebooks'), html.P('Repositorio Telco_Churn_Entregable')], href=REPO, target='_blank', className='res'),
             html.A([html.H4('🎛️ Este dashboard'), html.P('Repositorio Dash_Telco_Churn')], href=DASH_REPO, target='_blank', className='res'),
             html.A([html.H4('🗂️ Conjunto de datos'), html.P('Kaggle · Telco Customer Churn (IBM)')], href='https://www.kaggle.com/datasets/blastchar/telco-customer-churn', target='_blank', className='res'),
             html.A([html.H4('📄 Paper'), html.P('Versión web y PDF')], href=BOOK + 'paper.html', target='_blank', className='res'), cols=4)])


def pretty(feature):
    """cat__Contract_Two year → Contrato: Dos años; num__tenure → Antigüedad (meses)."""
    name = feature.split('__', 1)[-1]
    for var in sorted(a.LABELS, key=len, reverse=True):
        if name == var: return a.LABELS[var]
        if name.startswith(var + '_'): v = name[len(var) + 1:]; return f'{a.LABELS[var]}: {a.ES.get(v, v)}'
    return name


def chapters(chapter, pattern):
    chunks = [c for c in a.CHAPTERS[chapter] if re.search(pattern, c['title'], re.I)]
    return html.Details([html.Summary('📖 Leer el detalle del libro'), *[html.Details([html.Summary(c['title']), md(c['text'])], className='chapter') for c in chunks]], className='details')


PAGES = {'inicio': lambda d: cover(), **{k: globals()[k] for k, _, _ in NAV}}

# ---------------------------------------------------------------- layout
TICKER = ['📵 Contrato mes a mes: 42.7 % de abandono', '🌐 Fibra óptica: 42.1 %', '🧾 Cheque electrónico: 45.7 %', '⏳ Primer año: el periodo más frágil',
          '🛡️ Sin seguridad en línea: 41.9 %', '🏆 Mejor F1: 0.638 · Random Forest + SMOTE', '⚖️ SMOTE mejoró 6 de 7 familias', '🔍 Contrato, internet y antigüedad lideran SHAP']

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Header([
        dcc.Link([html.Span([html.B(), html.B(), html.B(), html.B()], className='logo-sig'), html.Span([html.B('Señal'), ' perdida'], className='logo-text')], href='#inicio', className='logo'),
        html.Nav([dcc.Link([html.Span(i, className='nav-i'), t], href='#' + k, id='nav-' + k, className='nav-link') for k, t, i in NAV], className='nav'),
        html.A('📘 Jupyter Book', href=BOOK, target='_blank', className='btn small')], className='topbar', id='topbar'),
    html.Div(html.Div([html.Span(t) for t in TICKER * 2], className='ticker-track'), className='ticker', id='ticker'),
    html.Div([field('Universo', dropdown('split', ['train', 'test', 'all'], 'train', {'train': 'Entrenamiento · 5,634', 'test': 'Test · 1,409', 'all': 'Todo · 7,043'})),
              field('Contrato', dropdown('contract', ['Todos'] + sorted(a.RAW.Contract.unique()), 'Todos')),
              field('Internet', dropdown('internet', ['Todos'] + sorted(a.RAW.InternetService.unique()), 'Todos')),
              field('Antigüedad (meses)', dcc.RangeSlider(0, 72, 1, value=[0, 72], marks={0: '0', 24: '24', 48: '48', 72: '72'}, id='tenure', tooltip={'placement': 'bottom'})),
              html.Button('↺ Restablecer', id='reset', className='btn ghost small')], className='filters', id='filters'),
    html.Main(id='content'),
    html.Footer([html.Span('Señal perdida · Telco Customer Churn'), html.Span('Laura Rivera & Natalý Cárdenas · Universidad del Norte'), html.A('Jupyter Book ↗', href=BOOK, target='_blank')],
                className='footer', id='footer'),
    dcc.Download(id='download')], id='shell')


@app.callback(Output('content', 'children'), *[Output('nav-' + k, 'className') for k, _, _ in NAV], Output('filters', 'style'), Output('shell', 'className'),
              Input('url', 'hash'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'))
def route(hash_, split, contract, internet, tenure):
    key = (hash_ or '#inicio').lstrip('#')
    if key not in PAGES: key = 'inicio'
    filters_changed = ctx.triggered_id in {'split', 'contract', 'internet', 'tenure'}
    if filters_changed and key not in DATA_PAGES:
        return (no_update,) * (len(NAV) + 3)
    d = a.cohort(split, contract, internet, tenure)
    return (html.Div(PAGES[key](d), className='page ' + key), *['nav-link on' if key == k else 'nav-link' for k, _, _ in NAV],
            {} if key in DATA_PAGES else {'display': 'none'}, 'shell cover-mode' if key == 'inicio' else 'shell')


@app.callback(Output('split', 'value'), Output('contract', 'value'), Output('internet', 'value'), Output('tenure', 'value'), Input('reset', 'n_clicks'), prevent_initial_call=True)
def reset(_): return 'train', 'Todos', 'Todos', [0, 72]


@app.callback(Output('var-chip-output', 'children'), Input('var-chip', 'value'))
def var_chip(variable):
    return bi(variable, 'train', 'Todos', 'Todos', [0, 72])


@app.callback(Output('uni-output', 'children'), Input('uni-var', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'))
def uni(variable, split, contract, internet, tenure):
    d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    if variable in a.NUMERIC:
        s = d[variable].dropna(); q = s.quantile([.25, .5, .75]); iqr = q.loc[.75] - q.loc[.25]; out = ((s < q.loc[.25] - 1.5 * iqr) | (s > q.loc[.75] + 1.5 * iqr)).sum()
        h = px.histogram(d, x=variable, color='Estado', color_discrete_map=COLORS, nbins=35, barmode='stack', marginal='box', labels={variable: a.LABELS[variable]})
        return html.Div([kpis(kpi('MEDIA', round(s.mean(), 2), '', '', 2), kpi('MEDIANA', round(s.median(), 2), '', '', 2), kpi('EXTREMOS IQR', int(out)), kpi('VACÍOS', int(d[variable].isna().sum()))),
                         chart('Distribución · ' + a.LABELS[variable], h, f'{len(s):,} valores válidos · asimetría {s.skew():.2f}. El color separa a quienes se van de quienes se quedan.', 420)])
    g = d.groupby([variable, 'Estado']).size().reset_index(name='Clientes'); g[variable] = g[variable].map(lambda v: a.ES.get(v, v)).astype(str)
    f = px.bar(g, x=variable, y='Clientes', color='Estado', color_discrete_map=COLORS, text_auto=True); f.update_xaxes(title='')
    return chart('Frecuencias · ' + a.LABELS[variable], f, f'{g[variable].nunique()} categorías en esta selección.', 400)


@app.callback(Output('bi-output', 'children'), Input('bi-var', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'))
def bi(variable, split, contract, internet, tenure):
    d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    if variable in a.NUMERIC:
        v = px.violin(d, x='Estado', y=variable, color='Estado', color_discrete_map=COLORS, box=True, points=False, labels={variable: a.LABELS[variable]})
        med = ', '.join(f'{k}: {m:.1f}' for k, m in d.groupby('Estado')[variable].median().items())
        return chart('¿Quién se va según ' + a.LABELS[variable].lower() + '?', v, 'Medianas · ' + med, 380, False)
    g = a.category_rates(d, variable); g['Categoría'] = g[variable].map(lambda x: a.ES.get(x, x)).astype(str); g = g.sort_values('tasa')
    f = px.bar(g, x='tasa', y='Categoría', orientation='h', text=g.tasa.map('{:.1%}'.format), custom_data=['n', 'abandonos'], color='tasa', color_continuous_scale=SEQ, range_color=[0, .6])
    f.update_xaxes(tickformat='.0%', title='Tasa de abandono'); f.update_yaxes(title=''); f.update_coloraxes(showscale=False)
    f.update_traces(hovertemplate='%{y}<br>Abandono: %{x:.1%}<br>n=%{customdata[0]} · abandonos=%{customdata[1]}<extra></extra>')
    top = g.iloc[-1]
    return chart('Tasa de abandono · ' + a.LABELS.get(variable, variable), f, f'Más frágil: {top["Categoría"]} ({top.tasa:.1%}, n={int(top.n):,}). Asociación descriptiva, no causal.', max(260, 70 + 48 * len(g)), False)


@app.callback(Output('scatter-output', 'children'), Input('scatter-x', 'value'), Input('scatter-y', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'))
def scatter(x, y, split, contract, internet, tenure):
    d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    sample = d.sample(min(2000, len(d)), random_state=42)
    return html.Div([graph(px.scatter(sample, x=x, y=y, color='Estado', color_discrete_map=COLORS, opacity=.55, labels=a.LABELS), 330),
                     html.P(f'Muestra de {len(sample):,} clientes · Spearman = {d[[x, y]].corr(method="spearman").iloc[0, 1]:.3f}' if x != y else 'Elige dos variables distintas.', className='insight')])


@app.callback(Output('models-output', 'children'), Input('model-family', 'value'), Input('model-balance', 'value'), Input('model-metric', 'value'))
def model_output(family, balance, metric):
    d = a.SUMMARY.copy()
    if family != 'Todas': d = d[d.model == family]
    if balance != 'Todos': d = d[d.balance == balance]
    d = d.sort_values(metric + '_mean', ascending=False); top = d.head(12).copy()
    top['Configuración'] = top.model.map(a.MODELS) + ' · ' + top.balance.map(a.BALANCES) + ' · ' + top.optimizer.map(a.OPTIMIZERS)
    f = px.bar(top.iloc[::-1], x=metric + '_mean', y='Configuración', error_x=metric + '_std', orientation='h', color=metric + '_mean', color_continuous_scale=[[0, VIOLET], [1, CYAN]],
               hover_data={metric + '_std': ':.3f'})
    lo = max(0, top[metric + '_mean'].min() - .08)
    f.update_xaxes(title=METRICS[metric] + ' · media externa', range=[lo, min(1, top[metric + '_mean'].max() + .05)]); f.update_yaxes(title=''); f.update_coloraxes(showscale=False)
    heat = px.imshow(d.pivot_table(index='model', columns='balance', values=metric + '_mean', aggfunc='mean').rename(index=a.MODELS, columns=a.BALANCES), text_auto='.3f',
                     color_continuous_scale=[[0, '#0b1426'], [.5, VIOLET], [1, CYAN]], aspect='auto', labels={'x': '', 'y': '', 'color': METRICS[metric]})
    view = d[['model', 'balance', 'optimizer', metric + '_mean', metric + '_std', 'recall_positive_mean', 'precision_positive_mean', 'roc_auc_mean']].copy()
    view = view.loc[:, ~view.columns.duplicated()]; view['model'] = view.model.map(a.MODELS); view['balance'] = view.balance.map(a.BALANCES); view['optimizer'] = view.optimizer.map(a.OPTIMIZERS)
    return html.Div([grid(chart(f'Top {len(top)} de {len(d)} configuraciones', f, 'Barras: media de 5 folds externos; bigotes: desviación estándar entre folds.', 480, False),
                          chart('Familia × balanceo', heat, 'Promedio sobre optimizadores. Las columnas con SMOTE brillan más en casi todas las familias.', 480, False)),
                     html.Details([html.Summary(f'📋 Tabla completa · {len(d)} configuraciones'), table(view)], className='details')])


@lru_cache(maxsize=1)
def anytime(): return pd.read_csv(a.DATA / 'anytime.csv')


@app.callback(Output('opt-output', 'children'), Input('opt-family', 'value'))
def opt_output(family):
    d = anytime(); d = d[(d.model == family) & (d.replicate == 0)]
    g = d.groupby(['optimizer', 'evaluation']).best_inner_f1.mean().reset_index(); g['Buscador'] = g.optimizer.map(a.OPTIMIZERS)
    frames = pd.concat([g[g.evaluation <= k].assign(Paso=k) for k in range(1, int(g.evaluation.max()) + 1)])
    lo, hi = g.best_inner_f1.min(), g.best_inner_f1.max(); pad = (hi - lo) * .15 + .002
    f = px.line(frames, x='evaluation', y='best_inner_f1', color='Buscador', animation_frame='Paso', markers=True, range_x=[0.5, g.evaluation.max() + .5], range_y=[lo - pad, hi + pad],
                labels={'evaluation': 'Evaluación', 'best_inner_f1': 'Mejor F1 interno'}, color_discrete_sequence=[CYAN, CORAL, VIOLET, AMBER])
    f.layout.updatemenus[0].buttons[0].args[1].update(frame={'duration': 220, 'redraw': False}, transition={'duration': 120})
    f.update(data=f.frames[-1].data); f.layout.sliders[0].active = len(f.frames) - 1   # arranca mostrando la carrera completa
    u = a.UNITS[(a.UNITS.model == family) & (a.UNITS.replicate == 0)].copy(); u['Buscador'] = u.optimizer.map(a.OPTIMIZERS)
    b = px.box(u, x='Buscador', y='search_seconds', color='Buscador', color_discrete_sequence=[CYAN, CORAL, VIOLET, AMBER], labels={'search_seconds': 'Segundos de búsqueda'})
    return grid(chart('▶ La carrera: mejor F1 acumulado', f, 'Dale play para ver cómo cada buscador mejora evaluación tras evaluación.', 430),
                chart('Tiempo de búsqueda', b, 'Segundos por unidad externa en el equipo del experimento.', 430, False))


@app.callback(Output('cal-output', 'children'), Input('cal-family', 'value'))
def cal_output(family):
    d = a.UNITS[(a.UNITS.model == family) & (a.UNITS.replicate == 0)]
    g = d.groupby('balance')[['brier', 'cal_brier', 'ece', 'cal_ece']].mean().reset_index(); g.balance = g.balance.map(a.BALANCES)
    if g.brier.isna().all(): return note('Sin probabilidad nativa', 'SVM se configuró sin probabilidades nativas; solo la rama calibrada tiene Brier y ECE.')
    m = g.melt(id_vars='balance', value_vars=['brier', 'cal_brier'], var_name='Rama', value_name='Brier'); m.Rama = m.Rama.map({'brier': 'Sin calibrar', 'cal_brier': 'Calibrado'})
    return html.Div([graph(px.bar(m, x='balance', y='Brier', color='Rama', barmode='group', color_discrete_sequence=[CORAL, MINT], labels={'balance': ''}), 300),
                     html.P('Menor es mejor. Promedio sobre folds externos y optimizadores.', className='insight')])


@app.callback(Output('sim-output', 'children'), Input('sim-contract', 'value'), Input('sim-internet', 'value'), Input('sim-payment', 'value'), Input('sim-tenure', 'value'), Input('sim-monthly', 'value'))
def sim_output(contract, internet, payment, tenure, monthly):
    s, w = a.similar_customers(contract, internet, payment, tenure, monthly)
    if s.empty: return card(note('Sin clientes parecidos', 'Prueba con otra combinación de contrato e internet.'))
    r = s.target.mean(); lv, col = level(r)
    ventana = f'antigüedad ±{w["tenure"]} meses · cargo ±${w["monthly"]}' + (' · mismo método de pago' if w['payment'] else '')
    return card(html.Div([html.Div([signal(r), html.Strong(f'Riesgo {lv}', style={'color': col}), html.Small(f'{len(s):,} clientes parecidos')], className='sim-sig'),
                          graph(gauge(r, 'Abandono observado en clientes similares', col), 260, False, margin=GAUGE_M)], className='sim-top'),
                html.P(['Criterio de similitud: mismo contrato e internet, ', ventana, '.'], className='insight'),
                html.Div([html.Span(f'{int(s.target.sum())} se fueron', className='pill coral'), html.Span(f'{int((1 - s.target).sum())} se quedaron', className='pill cyan')], className='pc-pills'))


@app.callback(Output('real-id', 'value'), Input('real-random', 'n_clicks'), prevent_initial_call=True)
def real_random(n): return int(np.random.default_rng(n).choice(a.TEST_IDS))


@app.callback(Output('real-output', 'children'), Input('real-id', 'value'))
def real_output(row):
    c = a.RAW.loc[row]; sc = a.test_scores().loc[row]
    facts = [('Contrato', a.ES[c.Contract]), ('Internet', a.ES.get(c.InternetService, c.InternetService)), ('Pago', a.ES[c.PaymentMethod]), ('Antigüedad', f'{c.tenure} meses'),
             ('Cargo mensual', f'${c.MonthlyCharges:.2f}'), ('Soporte técnico', a.ES[c.TechSupport]), ('Seguridad en línea', a.ES[c.OnlineSecurity]), ('Adulto mayor', 'Sí' if c.SeniorCitizen else 'No')]
    gauges = [graph(gauge(sc[e], a.FINALIST_NAMES[e], level(sc[e])[1]), 220, False, margin=GAUGE_M) for e in a.FINALISTS]
    verdict = html.Div(['📵 En la realidad este cliente ', html.B('abandonó')] if c.target else ['📶 En la realidad este cliente ', html.B('permaneció')], className='truth ' + ('coral' if c.target else 'cyan'))
    return html.Div([html.Div([html.Div([html.Div([html.Small('Cliente'), html.Strong(c.customerID)], className='pc-row'),
                                         html.Div([html.Div([html.Small(k), html.Strong(v)]) for k, v in facts], className='facts'), verdict], className='mini-phone'),
                               html.Div(gauges, className='gauges')], className='real'),
                     html.P('Probabilidades nativas guardadas de los modelos finalistas, sin recalibrar. El umbral por defecto es 0.5.', className='insight')])


@app.callback(Output('threshold-output', 'children'), Input('threshold-model', 'value'), Input('threshold', 'value'))
def threshold_output(experiment, threshold):
    p = a.predictions(experiment); cm, m = a.threshold_metrics(p, threshold)
    tn, fp, fn, tp = cm.ravel(); total = cm.sum()
    cells = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}; quota = {k: round(v / total * 100) for k, v in cells.items()}
    quota['tn'] += 100 - sum(quota.values())
    waffle = [html.Span(className='w ' + k, title=k) for k in ['tp', 'fn', 'fp', 'tn'] for _ in range(max(0, quota[k]))]
    legend = [('tp', 'Abandonos detectados', tp), ('fn', 'Abandonos que se escapan', fn), ('fp', 'Falsas alertas', fp), ('tn', 'Clientes estables bien leídos', tn)]
    fpr, tpr, thr = roc_curve(p.y_true, p.score)
    roc = go.Figure(go.Scatter(x=fpr, y=tpr, line_color=CYAN, name='ROC', fill='tozeroy', fillcolor='rgba(34,211,238,.08)'))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line={'dash': 'dash', 'color': '#475569'}, name='Azar'))
    roc.add_trace(go.Scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode='markers', marker={'size': 14, 'color': CORAL, 'line': {'color': 'white', 'width': 2}}, name='Umbral actual'))
    roc.update_xaxes(title='Tasa de falsas alertas'); roc.update_yaxes(title='Tasa de detección')
    return html.Div([kpis(*[kpi(k.upper(), round(v * 100, 1) if k != 'Alertas' else v, 'test histórico', 'coral' if k == 'Recall' else '', 1 if k != 'Alertas' else 0, ' %' if k != 'Alertas' else '')
                            for k, v in m.items()]),
                     grid(card(html.H3('100 clientes del test'), html.Div(waffle, className='waffle'),
                               html.Div([html.Div([html.Span(className='w ' + k), f'{t}: ', html.B(f'{v:,}')], className='leg') for k, t, v in legend], className='legend')),
                          chart('Curva ROC y punto de operación', roc, f'AUC = {roc_auc_score(p.y_true, p.score):.3f}. El punto rojo se mueve con el umbral.', 360))])


@app.callback(Output('explain-output', 'children'), Input('explain-model', 'value'))
def explain_output(experiment):
    path = a.DATA / 'interpretabilidad' / experiment
    g = pd.read_csv(path / 'importancia_global.csv').rename(columns={'Unnamed: 0': 'Variable'}).head(10)
    g['Nombre'] = g.Variable.map(lambda v: a.LABELS.get(v, v))
    fg = px.bar(g.iloc[::-1], x='mean_abs_shap', y='Nombre', orientation='h', color='mean_abs_shap', color_continuous_scale=[[0, VIOLET], [1, CORAL]], text=g.iloc[::-1].mean_abs_shap.map('{:.3f}'.format))
    fg.update_coloraxes(showscale=False); fg.update_yaxes(title=''); fg.update_xaxes(title='Media |SHAP|')
    ids = [4267, 4386] if experiment == a.FINALISTS[0] else [2221, 4513]
    local = []
    for row in ids:
        s = pd.read_csv(path / f'shap_local_{row}.csv'); s = s.reindex(s.shap.abs().sort_values(ascending=False).index).head(8).sort_values('shap')
        s['Efecto'] = np.where(s.shap > 0, 'Empuja a irse', 'Empuja a quedarse'); s['feature'] = s.feature.map(pretty)
        pred = a.predictions(experiment).set_index('row_id').loc[row]
        ok = pred.y_true == pred.y_pred
        f = px.bar(s, x='shap', y='feature', orientation='h', color='Efecto', color_discrete_map={'Empuja a irse': CORAL, 'Empuja a quedarse': CYAN}); f.update_yaxes(title=''); f.update_xaxes(title='SHAP')
        local.append(card(html.Div([html.H3(f'Cliente {a.RAW.customerID[row]}'), html.Span('✅ acierto' if ok else '❌ error', className='pill ' + ('cyan' if ok else 'coral'))], className='card-head'),
                          html.P(f'Realidad: {"abandonó" if pred.y_true else "permaneció"} · probabilidad del modelo: {pred.probability:.1%}', className='muted'), graph(f, 320),
                          html.Details([html.Summary('Ver explicación LIME'), html.Img(src=f'/evidence/{experiment}/lime_local_{row}.png', className='evidence', alt=f'LIME del cliente {row}')], className='details')))
    return html.Div([chart('Importancia global · ¿qué mira el modelo?', fg, 'Promedio de |SHAP| en los 5,634 clientes de desarrollo. Mide cuánto pesa cada variable, no en qué dirección causal.', 400, False),
                     html.H3('Dos clientes bajo la lupa', className='section-title'), grid(*local)])


@app.callback(Output('explorer-table', 'children'), Input('explorer-cols', 'value'), Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value'))
def explorer_table(cols, split, contract, internet, tenure):
    d = a.cohort(split, contract, internet, tenure)
    return table(d[cols or ['customerID']], id='data-table', page_size=12) if not d.empty else empty()


@app.callback(Output('download', 'data'), Input('export', 'n_clicks'), State('explorer-cols', 'value'), State('split', 'value'), State('contract', 'value'), State('internet', 'value'), State('tenure', 'value'), prevent_initial_call=True)
def export(_, cols, split, contract, internet, tenure): return dcc.send_data_frame(a.cohort(split, contract, internet, tenure)[cols or ['customerID']].to_csv, 'telco_seleccion.csv', index=False)


@app.callback(Output('reader-output', 'children'), Input('reader-chapter', 'value'), Input('reader-search', 'value'))
def reader_output(chapter, search):
    search = (search or '').casefold(); chunks = [c for c in a.CHAPTERS[chapter] if search in c['text'].casefold() or search in c['title'].casefold()]
    return html.Div([html.P(f'{len(chunks)} secciones encontradas', className='muted'), *[html.Details([html.Summary(c['title']), md(c['text'])], className='chapter') for c in chunks]])


app.clientside_callback("function(h){window.scrollTo({top:0,behavior:'smooth'});var k=(h||'#inicio').slice(1);var t=(" + json.dumps(TITLES) + ")[k]||'Inicio';document.title=t+' · Señal perdida';return '';}",
                        Output('footer', 'title'), Input('url', 'hash'))

if __name__ == '__main__': app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=False)
