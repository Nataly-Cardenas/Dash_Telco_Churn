"""Sección 2 · EDA: filtros globales, distribución del objetivo, univariado, bivariado, correlación, faltantes y explorador."""
import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update
from sklearn.metrics import confusion_matrix, roc_curve

from src import data as a
from src import eda
from src.i18n import N, T, set_lang
from src.icons import icon
from src.plots import BASE, DIVERGING, GREY, HOT, SEQ, go, highlight, make_subplots, px, roc_animation
from src.ui import (card, cat, chart, churn_label, details, dropdown, empty, field, graph, grid, kpi, kpis, note, options,
                    page_head, scope, subhead, table, tabs, tt)

dash.register_page(__name__, path='/eda', name='EDA', title='EDA · Señal perdida', order=1)

FILTERS = [Input('split', 'value'), Input('contract', 'value'), Input('internet', 'value'), Input('tenure', 'value')]
EXPLORABLES = a.NUMERIC + ['SeniorCitizen']


def outcome_colors(): return {T('Permanece'): BASE, T('Abandona'): HOT}


# ---------------------------------------------------------------- layout
def layout(**_):
    return html.Div([
        html.Div([field('', dropdown('split', ['train', 'test', 'all'], 'train'), id='f-split'),
                  field('', dropdown('contract', ['Todos'] + sorted(a.RAW.Contract.unique()), 'Todos'), id='f-contract'),
                  field('', dropdown('internet', ['Todos'] + sorted(a.RAW.InternetService.unique()), 'Todos'), id='f-internet'),
                  field('', dcc.RangeSlider(0, 72, 1, value=[0, 72], marks={0: '0', 24: '24', 48: '48', 72: '72'}, id='tenure', tooltip={'placement': 'bottom'}), id='f-tenure'),
                  html.Button([icon('reset', 'sm'), html.Span(id='reset-t')], id='reset', className='btn ghost small')], className='filters', id='filters'),
        dcc.Loading(html.Div(id='eda-contenido', className='page eda'), type='dot', color=BASE, delay_show=400)])


@callback(Output('f-split', 'children'), Output('f-contract', 'children'), Output('f-internet', 'children'), Output('f-tenure', 'children'), Output('reset-t', 'children'),
          Output('split', 'options'), Output('contract', 'options'), Output('internet', 'options'), Input('lang', 'data'))
def filtros(lang):
    set_lang(lang)
    return (T('Universo'), T('Contrato'), T('Internet'), T('Antigüedad (meses)'), T('Restablecer'),
            options(['train', 'test', 'all'], {'train': T('Entrenamiento · 5,634'), 'test': T('Test · 1,409'), 'all': T('Todo · 7,043')}),
            options(['Todos'] + sorted(a.RAW.Contract.unique()), {'Todos': T('Todos')}), options(['Todos'] + sorted(a.RAW.InternetService.unique()), {'Todos': T('Todos')}))


@callback(Output('split', 'value'), Output('contract', 'value'), Output('internet', 'value'), Output('tenure', 'value'), Input('reset', 'n_clicks'), prevent_initial_call=True)
def reset(_): return 'train', 'Todos', 'Todos', [0, 72]


@callback(Output('eda-contenido', 'children'), Input('lang', 'data'), *FILTERS)
def render(lang, split, contract, internet, tenure):
    set_lang(lang)
    return eda_page(a.cohort(split, contract, internet, tenure))


def eda_page(d):
    if d.empty: return html.Div([page_head(T('Análisis exploratorio'), T('Explora el conjunto de datos con los filtros superiores.'), 'EDA_'), empty()])
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
    color_opts = {'Churn': T('Abandono'), **{v: v for v in a.CATEGORICAL}}
    explorador_xy = [card(html.Div([field(T('Eje X'), dropdown('xy-x', EXPLORABLES + a.CATEGORICAL, 'tenure', {v: v for v in EXPLORABLES + a.CATEGORICAL})),
                                    field(T('Eje Y'), dropdown('xy-y', EXPLORABLES, 'MonthlyCharges', {v: v for v in EXPLORABLES})),
                                    field(T('Color'), dropdown('xy-color', ['Churn'] + a.CATEGORICAL, 'Churn', color_opts))], className='controls'),
                         html.Div(id='xy-output'), cls='controls-card')]
    explorador = [card(field(T('Columnas visibles'), dropdown('explorer-cols', list(a.RAW.columns[:21]), ['customerID', 'Contract', 'InternetService', 'tenure', 'MonthlyCharges', 'Churn'], {}, multi=True)),
                       html.Div(id='explorer-table'),
                       html.Div([html.Button([icon('download'), T('Descargar CSV')], id='export', className='btn primary', n_clicks=0),
                                 html.P(T('Descarga un solo archivo con la selección actual de los filtros superiores y las columnas elegidas.'), className='muted')], className='export-row'), cls='controls-card')]
    return html.Div([page_head(T('Análisis exploratorio'), T('Del archivo original a los patrones de abandono. Todo el análisis se hace sobre el conjunto de entrenamiento; los filtros superiores recalculan las gráficas y tablas.'), 'EDA_'),
                     scope(d),
                     kpis(kpi(T('CLIENTES'), len(d), T('en la selección')), kpi(T('ABANDONO'), round(d.target.mean() * 100, 2), T('{n} clientes').format(n=f'{int(d.target.sum()):,}'), 'hot', 2, ' %'),
                          kpi(T('ANTIGÜEDAD MEDIA'), round(d.tenure.mean(), 1), T('meses'), '', 1), kpi(T('CARGO MENSUAL MEDIO'), round(d.MonthlyCharges.mean(), 2), T('dólares'), '', 2, '', '$')),
                     note(T('Sin fechas en los datos'), T('El conjunto es una fotografía de corte transversal: no hay una variable de fecha para filtrar por periodo. Los filtros usan partición, contrato, internet y antigüedad.')),
                     tabs('eda-tabs', [(T('1–2 · ETL y partición'), etl), (T('3 · Univariado'), univariado), (T('4 · Bivariado'), bivariado),
                                       (T('Explorador bivariado'), explorador_xy), (T('5 · Multivariado'), [html.Div(id='multi-output')]),
                                       (T('6 · Valores faltantes'), [html.Div(id='missing-output')]), (T('7–8 · Resumen'), [eda_summary()]),
                                       (T('Preparación de datos'), preparacion()), (T('Explorador de datos'), explorador)])])


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


# ---------------------------------------------------------------- preparación de datos (capítulo de preprocesamiento)
STEPS = [('target', N('4.1 Objetivo'), N('Churn Yes/No → 1/0. customerID solo para control.')), ('shield', N('4.2 TotalCharges'), N('Texto a número; los 8 vacíos de train (3 de test) tienen tenure 0 → 0.')),
         ('layers', N('4.3 Ingeniería'), N('“Sin internet / sin teléfono” → “No” y AvgChargePerMonth = TotalCharges / tenure.')), ('table', N('4.4–4.5 Codificar y escalar'), N('5 numéricas con StandardScaler y 15 categóricas con One-Hot: 24 columnas.')),
         ('scale', N('Calibrar'), N('Platt (sigmoide, 3 pliegues) dentro del entrenamiento de cada modelo; sin remuestreo.')), ('network', N('Modelo'), N('Se ajusta en entrenamiento y se evalúa en el pliegue reservado.'))]


def preparacion():
    imput = pd.DataFrame({'m': [T('Imputar con 0'), T('Mediana + indicador')], 'auc': [.8459, .8460], 'sd': [.0124, .0125]})
    fi = go.Figure(go.Bar(x=imput.m, y=imput.auc, error_y={'type': 'data', 'array': imput.sd, 'color': GREY}, marker_color=[BASE, GREY], text=imput.auc.map('{:.4f}'.format), textposition='inside'))
    fi.update_yaxes(range=[.8, .87], title='AUC-ROC CV')
    b = pd.read_csv(a.DATA / 'baseline_test.csv'); fpr, tpr, _ = roc_curve(b.y_true, b.probability)
    cm = confusion_matrix(b.y_true, (b.probability >= .5).astype(int))
    fcm = px.imshow(cm, x=[T('Predice permanencia'), T('Predice abandono')], y=[T('Permanece'), T('Abandona')], text_auto=True, color_continuous_scale=SEQ, aspect='auto')
    fcm.update_coloraxes(showscale=False); fcm.update_traces(textfont_size=18)
    return [card(html.H3(T('El pipeline paso a paso'), className='chart-title'), html.Div([html.Div(className='flow-line'), *[html.Div([html.Span(icon(i), className='st-icon'), html.H4(T(t)), html.P(T(s))], className='station',
                 style={'animationDelay': f'{k * .15}s'}) for k, (i, t, s) in enumerate(STEPS)]], className='pipeline')),
            grid(chart(T('4.2 ¿Cero o mediana para TotalCharges?'), fi, T('Diferencia pareada por fold: +0.0001 ± 0.0001. Imputar con 0 es coherente con clientes sin facturación y no pierde desempeño.'), 320, False),
                 card(html.H3(T('4.3 Ingeniería de características'), className='chart-title'),
                      kpis(kpi(T('CORRELACIÓN CON MONTHLYCHARGES'), .9963, 'AvgChargePerMonth', '', 4), kpi(T('COLUMNAS TRAS CODIFICAR'), 24, T('5 numéricas + 19 indicadores')), cls='k2'),
                      html.P(T('El cargo promedio mensual comparte casi toda la información con el cargo mensual; su aporte se evalúa en la ablación.'), className='insight'))),
            grid(chart(T('Modelo base del entregable 2 · matriz de confusión en test'), fcm, T('Regresión logística con umbral 0.5: detecta 293 de 374 abandonos, con 286 falsas alertas.'), 340, False),
                 chart(T('Modelo base del entregable 2 · curva ROC en test'), roc_animation([(T('Regresión logística'), fpr, tpr, BASE)]), T('Pulsa play para trazar la curva. AUC = 0.8419.'), 340))]


# ---------------------------------------------------------------- callbacks del EDA
@callback(Output('uni-num-output', 'children'), Input('uni-num', 'value'), *FILTERS, State('lang', 'data'))
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


@callback(Output('uni-cat-output', 'children'), Input('uni-cat', 'value'), *FILTERS, State('lang', 'data'))
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


@callback(Output('bi-num-output', 'children'), Output('bi-pb-output', 'children'), Input('bi-num', 'value'), *FILTERS, State('lang', 'data'))
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


@callback(Output('bi-cat-output', 'children'), Output('bi-holm-output', 'children'), Input('bi-cat', 'value'), *FILTERS, State('lang', 'data'))
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


@callback(Output('xy-output', 'children'), Input('xy-x', 'value'), Input('xy-y', 'value'), Input('xy-color', 'value'), *FILTERS, State('lang', 'data'))
def explorador_xy(x, y, color, split, contract, internet, tenure, lang='es'):
    """Explorador bivariado: el usuario elige los ejes y la variable de color."""
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    if d.empty: return empty()
    if x == y: return note(T('Elige dos variables distintas'), T('El eje X y el eje Y muestran la misma variable.'))
    s = d.sample(min(2000, len(d)), random_state=42).copy()
    s['_color'] = s.Churn.map(churn_label) if color == 'Churn' else s[color].map(cat).astype(str)
    niveles = sorted(s['_color'].unique(), key=str)
    paleta = outcome_colors() if color == 'Churn' else {k: c for k, c in zip(niveles, [BASE, HOT, GREY] * 10)}
    simbolos = {k: m for k, m in zip(niveles, ['circle', 'diamond', 'square', 'x', 'triangle-up'] * 4)}
    if x in a.CATEGORICAL:
        f = px.box(s, x=x, y=y, color='_color', color_discrete_map=paleta, points=False, labels={'_color': '', x: x, y: y})
        f.update_traces(quartilemethod='linear'); f.update_xaxes(tickvals=sorted(s[x].unique()), ticktext=[cat(v) for v in sorted(s[x].unique())])
        insight = T('Cajas de {y} por categoría de {x}, separadas por {c}.').format(y=y, x=x, c=T('abandono') if color == 'Churn' else color)
    else:
        f = px.scatter(s, x=x, y=y, color='_color', symbol='_color', color_discrete_map=paleta, symbol_map=simbolos, opacity=.5, render_mode='webgl', labels={'_color': '', x: x, y: y})
        f.update_traces(marker_size=5)
        rho = s[[x, y]].corr(method='spearman').iloc[0, 1]
        insight = T('Muestra de hasta 2,000 clientes de la selección. Spearman entre {x} y {y}: ρ = {r}.').format(x=x, y=y, r=f'{rho:.2f}')
    if len(niveles) > 3 and color != 'Churn':
        insight += ' ' + T('Con más de tres categorías, los símbolos distinguen los grupos que comparten color.')
    return html.Div([graph(f, 460, True), html.P(insight, className='insight')])


@callback(Output('multi-output', 'children'), *FILTERS, State('lang', 'data'))
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


@callback(Output('missing-output', 'children'), *FILTERS, State('lang', 'data'))
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


@callback(Output('explorer-table', 'children'), Input('explorer-cols', 'value'), *FILTERS, State('lang', 'data'))
def explorer_table(cols, split, contract, internet, tenure, lang='es'):
    set_lang(lang); d = a.cohort(split, contract, internet, tenure)
    return table(d[cols or ['customerID']], id='data-table', page_size=12) if not d.empty else empty()


@callback(Output('download', 'data'), Input('export', 'n_clicks'), State('explorer-cols', 'value'), State('split', 'value'), State('contract', 'value'), State('internet', 'value'), State('tenure', 'value'), prevent_initial_call=True)
def export(n, cols, split, contract, internet, tenure):
    # Solo un clic real descarga: al volver a pintar la sección el botón nace con n_clicks = 0.
    if not n: return no_update
    return dcc.send_data_frame(a.cohort(split, contract, internet, tenure)[cols or ['customerID']].to_csv, 'telco_seleccion.csv', index=False)
