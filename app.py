"""Señal perdida: dashboard del proyecto Telco Customer Churn de Laura Rivera y Natalý Cárdenas.

Punto de entrada. Crea la aplicación multipágina (use_pages=True), la barra de navegación fija, el idioma y el tema,
y expone `server` para gunicorn. Cada sección vive en pages/; la lógica de datos y de modelos, en src/.
"""
import json
import os

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, ctx, dcc, html
from flask import abort, send_from_directory

from src import chatbot
from src import data as a
from src import modelos as M
from src.i18n import EN, N, T, set_lang
from src.icons import icon
from src.ui import BOOK

NAV = [('/', N('Contexto'), 'signal'), ('/eda', N('EDA'), 'chart'), ('/modelos', N('Modelos'), 'trophy')]
TICKER = [N('Contrato mes a mes: 42.7 % de abandono'), N('Fibra óptica: 42.1 %'), N('Cheque electrónico: 45.7 %'), N('Primer año: el periodo más frágil'),
          N('RegionalMAPEM: log-loss 0,41127'), N('Logística con splines: 0,41130'), N('Empate: ninguna prueba rechaza H0 tras Holm'), N('Contrato, antigüedad e internet lideran SHAP'),
          N('LinearSVC: 36× más rápido que SVC RBF'), N('Successive Halving: 7.1× menos tiempo en SVM')]
BOOTSTRAP_GRID = 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap-grid.min.css'
FONTS = 'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap'

app = Dash(__name__, use_pages=True, pages_folder='pages', suppress_callback_exceptions=True, title='Señal perdida · Telco Customer Churn',
           external_stylesheets=[BOOTSTRAP_GRID, FONTS],
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
                      {'name': 'description', 'content': 'Dashboard interactivo para entender y anticipar el abandono de clientes de telecomunicaciones.'}])
server = app.server
# El tema se aplica antes de pintar para evitar un destello del tema equivocado.
app.index_string = app.index_string.replace('<html>', '<html lang="es">').replace('{%favicon%}', '{%favicon%}<script>try{var t=JSON.parse(localStorage.getItem("theme"));if(t)document.documentElement.dataset.theme=t;var l=JSON.parse(localStorage.getItem("lang"));if(l)document.documentElement.lang=l;}catch(e){}</script>')


@server.route('/evidence/<path:filename>')
def evidence(filename):
    return send_from_directory(a.DATA / 'interpretabilidad', filename)


@server.route('/artifacts/<modelo>/metadata.json')
def artifact_metadata(modelo):
    if modelo not in M.MODELOS: abort(404)
    return send_from_directory(M.ART / modelo, 'metadata.json', mimetype='application/json')


@server.route('/healthz')
def health():
    return {'status': 'ok', 'clients': len(a.RAW), 'models': len(M.disponibles())}


def ticker():
    return html.Div([html.Span([html.I(className='tick-dot'), T(t)]) for t in TICKER * 2], className='ticker-track')


app.layout = html.Div([
    dcc.Location(id='url'), dcc.Store(id='lang', storage_type='local', data='es'), dcc.Store(id='theme', storage_type='local', data='dark'),
    html.Header([
        dcc.Link([html.Span([html.B(), html.B(), html.B(), html.B()], className='logo-sig'), html.Span(id='logo-text', className='logo-text')], href='/', className='logo'),
        html.Nav([dcc.Link([icon(i, 'nav-i'), html.Span(id='nav-t' + p.replace('/', '-'))], href=p, id='nav' + p.replace('/', '-'), className='nav-link') for p, _, i in NAV], className='nav'),
        html.Div([html.Div([html.Button('ES', id='lang-es', className='seg', n_clicks=0), html.Button('EN', id='lang-en', className='seg', n_clicks=0)], className='segmented', id='lang-switch'),
                  html.Button([icon('sun', 'sun'), icon('moon', 'moon')], id='theme-btn', className='icon-btn', n_clicks=0),
                  html.A([icon('book', 'sm'), html.Span('Jupyter Book', className='book-btn-t')], href=BOOK, target='_blank', className='btn small')], className='top-actions')], className='topbar', id='topbar'),
    html.Div(id='ticker', className='ticker'),
    html.Main(dash.page_container, id='content'),
    html.Footer(id='footer', className='footer'),
    chatbot.layout(),
    dcc.Download(id='download')], id='shell', className='shell')


@app.callback(Output('logo-text', 'children'), *[Output('nav-t' + p.replace('/', '-'), 'children') for p, _, _ in NAV], Output('ticker', 'children'), Output('footer', 'children'),
              Output('lang-es', 'className'), Output('lang-en', 'className'), Output('theme-btn', 'title'), Input('lang', 'data'))
def chrome(lang):
    """Textos fijos de la interfaz: se recalculan solo al cambiar de idioma."""
    set_lang(lang)
    footer = [html.Span(T('Señal perdida · Telco Customer Churn')), html.Span('Laura Rivera & Natalý Cárdenas · Universidad del Norte')]
    return ([html.B(T('Señal')), ' ' + T('perdida')], *[T(t) for _, t, _ in NAV], ticker(), footer,
            'seg on' if lang != 'en' else 'seg', 'seg on' if lang == 'en' else 'seg', T('Cambiar tema claro / oscuro'))


@app.callback(*[Output('nav' + p.replace('/', '-'), 'className') for p, _, _ in NAV], Input('url', 'pathname'))
def active(path):
    return ['nav-link on' if (path or '/') == p else 'nav-link' for p, _, _ in NAV]


@app.callback(Output('lang', 'data'), Input('lang-es', 'n_clicks'), Input('lang-en', 'n_clicks'), prevent_initial_call=True)
def choose_lang(_es, _en): return 'en' if ctx.triggered_id == 'lang-en' else 'es'


chatbot.register(app)

# Tema e idioma en el navegador: se guardan en localStorage (dcc.Store) y se aplican al <html>.
app.clientside_callback("function(n, t){ if(!n) return window.dash_clientside.no_update; return t === 'light' ? 'dark' : 'light'; }",
                        Output('theme', 'data'), Input('theme-btn', 'n_clicks'), State('theme', 'data'), prevent_initial_call=True)
app.clientside_callback("function(t){ document.documentElement.dataset.theme = t || 'dark'; return t || 'dark'; }", Output('theme-btn', 'value'), Input('theme', 'data'))
TITLES = {p: t for p, t, _ in NAV}
app.clientside_callback("function(p, l){window.scrollTo({top:0,behavior:'smooth'});document.documentElement.lang=l||'es';"
                        "var T=" + json.dumps({'es': TITLES, 'en': {k: EN.get(v, v) for k, v in TITLES.items()}}) + "[l==='en'?'en':'es'];"
                        "document.title=(T[p]||T['/'])+(l==='en'?' · Lost signal':' · Señal perdida');return '';}",
                        Output('footer', 'title'), Input('url', 'pathname'), Input('lang', 'data'))

if __name__ == '__main__': app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=False)
