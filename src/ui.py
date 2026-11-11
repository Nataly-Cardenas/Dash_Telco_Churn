"""Componentes de interfaz compartidos por las tres páginas (tarjetas, KPI, tablas, pestañas y avisos)."""
import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html, dash_table

from src import data as a
from src.i18n import T, get_lang
from src.icons import icon
from src.plots import style

BOOK = 'https://nataly-cardenas.github.io/Telco_Churn_JBook/'
REPO = 'https://github.com/Nataly-Cardenas/Telco_Churn_JBook'
DASH_REPO = 'https://github.com/Nataly-Cardenas/Dash_Telco_Churn'
VERSION = '4.0'
ACTUALIZADO = '2026-11-10'


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
def empty(): return note(T('Sin clientes en este segmento'), T('Amplía los filtros para volver a calcular las visualizaciones.'))
def scope(d): return html.Div([html.Span(className='live-dot'), ' ' + T('{n} clientes en la selección').format(n=f'{len(d):,}')], className='scope')


def alert(title, text, kind='warning'):
    """Mensaje de error o advertencia (dbc.Alert) con el estilo del tablero."""
    return dbc.Alert([html.Strong(title), html.P(text)], color=kind, className='alert-dash ' + kind, is_open=True)


def tt(d):
    """Traduce encabezados y textos de una tabla."""
    d = d.copy(); d.columns = [T(str(c)) for c in d.columns]
    for c in d.columns:
        if d[c].dtype == object: d[c] = d[c].map(lambda v: T(v) if isinstance(v, str) else v)
    return d


def table(d, id=None, **kw):
    args = {'id': id} if id else {}
    if kw.pop('wrap', False): args['style_data'] = {'whiteSpace': 'normal', 'height': 'auto'}
    d = d.copy()
    for c in d.columns:
        if d[c].dtype == bool: d[c] = d[c].map({True: T('Sí'), False: 'No'})
    return dash_table.DataTable(data=d.round(kw.pop('decimals', 4)).replace({np.nan: None}).to_dict('records'), columns=kw.pop('columns', None) or [{'name': str(c), 'id': str(c)} for c in d.columns],
                                page_size=kw.pop('page_size', 10), sort_action='native', filter_action=kw.pop('filter_action', 'none'), style_table={'overflowX': 'auto'},
                                style_header={'backgroundColor': 'var(--table-head)', 'color': 'var(--text)', 'border': 'none', 'fontWeight': '600', 'whiteSpace': 'normal'},
                                style_cell={'backgroundColor': 'var(--table-cell)', 'color': 'var(--soft)', 'border': 'none', 'borderBottom': '1px solid var(--line)', 'padding': '10px',
                                            'textAlign': 'left', 'fontFamily': 'Inter, sans-serif', 'fontSize': 13.5, 'maxWidth': 340, 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                                style_filter={'backgroundColor': 'var(--table-head)', 'color': 'var(--text)'}, **args, **kw)


def graph(f, height=360, legend=True, margin=None, id=None):
    args = {'id': id} if id else {}
    fit = f'min({height}px, calc(100vh - 230px))'   # nunca más alta que la pantalla disponible bajo la barra superior
    return dcc.Graph(figure=style(f, height, legend, margin), style={'height': fit, 'minHeight': f'{min(height, 240)}px', 'width': '100%'},
                     config={'displaylogo': False, 'responsive': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}, **args)


def chart(title, f, insight='', height=360, legend=True):
    return card(html.H3(title, className='chart-title'), graph(f, height, legend), html.P(insight, className='insight') if insight else None)


def page_head(title, lead, chapter=None):
    """Encabezado de página con un único enlace al capítulo del Jupyter Book."""
    link = html.A([icon('book', 'sm'), T('Ver este capítulo en el Jupyter Book'), icon('arrow', 'sm')], href=BOOK + chapter + '.html', target='_blank', className='chip-link') if chapter else None
    return html.Div([eyebrow('TELCO CUSTOMER CHURN'), html.H1(title), html.P(lead, className='lead'), link], className='page-head')


def tabs(id, items, value='0'):
    return dcc.Tabs([dcc.Tab(children=html.Div(content, className='tab-body'), label=lbl, value=str(i), className='tab', selected_className='tab on')
                     for i, (lbl, content) in enumerate(items)], value=value, id=id, className='tabs', parent_className='tabs-wrap', mobile_breakpoint=0)
