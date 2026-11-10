"""Resultados del entregable 2 (112 configuraciones): se conservan como pestaña de la sección Modelos.

Corresponden al capítulo MODELADO_AVANZADO del Jupyter Book. Usan predicciones y tablas guardadas; no cargan modelos.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src import data as a
from src.i18n import N, T, set_lang
from src.icons import icon
from src.plots import BASE, GREY, HOT, SEQ, go, highlight, make_subplots, px, roc_animation
from src.ui import (card, cat, chart, config_name, counter, details, dropdown, eyebrow, field, graph, grid, kpi, kpis, label,
                    model_name, balance_name, optimizer_name, note, subhead, table, tabs, tt)


@lru_cache(maxsize=1)
def oof(): return pd.read_csv(a.DATA / 'oof_mejores.csv')
@lru_cache(maxsize=1)
def anytime(): return pd.read_csv(a.DATA / 'anytime.csv')
@lru_cache(maxsize=1)
def diversity(): return pd.read_csv(a.DATA / 'diversidad_generaciones.csv')
@lru_cache(maxsize=1)
def shap_sample(): return pd.read_csv(a.DATA / 'shap_muestra.csv')


BEST_BY_FAMILY = {'knn': 'knn__smote__bayesian_tpe', 'naive_bayes': 'naive_bayes__none__random', 'logistic': 'logistic__smote__grid', 'decision_tree': 'decision_tree__smote__bayesian_tpe',
                  'random_forest': 'random_forest__smote__bayesian_tpe', 'xgboost': 'xgboost__smote__random', 'svm': 'svm__smote__genetic'}

METRICS = {'f1_positive': N('F1 positivo'), 'recall_positive': 'Recall', 'precision_positive': 'Precision', 'roc_auc': 'ROC-AUC', 'accuracy': 'Accuracy'}


def podium():
    p = [('2', 'XGBoost', 'SMOTE · Random', .638, 'second'), ('1', 'Random Forest', 'SMOTE · TPE', .638, 'first'), ('3', T('Reg. logística'), 'SMOTE · Grid', .634, 'third')]
    return card(eyebrow(T('PODIO · F1 MEDIO EXTERNO')), html.Div([html.Div([html.Span(m, className='medal'), html.Strong(n), html.Small(c), html.Div(counter(v, 3), className='bar-podium')],
                className='pod ' + k) for m, n, c, v, k in p], className='podium'), html.P(T('Empate técnico: la diferencia entre RF y XGBoost es 0.0004, menor que la variación entre folds.'), className='insight'))


def entregable2():
    fam = {k: model_name(k) for k in a.MODELS}
    resultados = [card(field(T('Familia de modelos (mejor configuración de cada una)'), dropdown('res-family', list(BEST_BY_FAMILY), 'random_forest', fam)), html.Div(id='res-output'), cls='controls-card'),
                  html.Div(id='rank-output-static', children=ranking_families()),
                  card(html.Div([field(T('Familia'), dropdown('model-family', ['Todas'] + list(a.MODELS), 'Todas', {'Todas': T('Todas'), **fam})),
                                 field(T('Balanceo'), dropdown('model-balance', ['Todos'] + list(a.BALANCES), 'Todos', {'Todos': T('Todos'), **{k: balance_name(k) for k in a.BALANCES}})),
                                 field(T('Métrica'), dropdown('model-metric', list(METRICS), 'f1_positive', {k: T(v) for k, v in METRICS.items()}))], className='controls'), cls='controls-card'),
                  html.Div(id='models-output')]
    calibracion = [html.Div(id='cal-output-static', children=calibration_summary())]
    optimizadores = [card(field(T('Familia'), dropdown('opt-family', list(a.MODELS), 'random_forest', fam)), cls='controls-card'), html.Div(id='opt-output')]
    return html.Div([note(T('Entregable 2'), T('Siete familias × cuatro balanceos × cuatro optimizadores = 112 configuraciones evaluadas con validación cruzada anidada y F1 como métrica. Se conservan como referencia; el modelo propio se comparó después con 11 benchmarks recalibrados.')),
                     html.Div([podium(), kpis(kpi(T('CONFIGURACIONES'), 112, '7 × 4 × 4'), kpi(T('UNIDADES EXTERNAS'), 1680, T('112 × 3 semillas × 5 folds')),
                                              kpi('FOLDS', '5 × 3', T('externos × internos')), kpi(T('PRESUPUESTO'), 24, T('evaluaciones por búsqueda')), cls='k2')], className='hero'),
                     tabs('model-tabs', [(T('1 · Resultados por modelo'), resultados), (T('1.8 · Calibración'), calibracion), (T('2 · Optimizadores'), optimizadores),
                                         (T('3 · Comparación estadística'), estadistica()), (T('3.2 · Interpretabilidad'), interpretabilidad()), (T('Umbral de alerta'), umbral_lab())])])


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

def pretty(feature):
    """cat__Contract_Two year → Contrato: Dos años; num__tenure → Antigüedad (meses)."""
    name = feature.split('__', 1)[-1]
    for var in sorted(a.LABELS, key=len, reverse=True):
        if name == var: return label(var)
        if name.startswith(var + '_'): return f'{label(var)}: {cat(name[len(var) + 1:])}'
    return name

@callback(Output('res-output', 'children'), Input('res-family', 'value'), State('lang', 'data'))
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


@callback(Output('models-output', 'children'), Input('model-family', 'value'), Input('model-balance', 'value'), Input('model-metric', 'value'), State('lang', 'data'))
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


@callback(Output('opt-output', 'children'), Input('opt-family', 'value'), State('lang', 'data'))
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


@callback(Output('explain-output', 'children'), Input('explain-model', 'value'), State('lang', 'data'))
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


@callback(Output('threshold-output', 'children'), Input('threshold-model', 'value'), Input('threshold', 'value'), State('lang', 'data'))
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

def umbral_lab():
    return [card(html.Div([field(T('Modelo finalista'), dropdown('threshold-model', a.FINALISTS, a.FINALISTS[0], {k: T(v) for k, v in a.FINALIST_NAMES.items()})),
                           field(T('Umbral de alerta'), dcc.Slider(.05, .95, .01, value=.5, marks={.1: '0.10', .3: '0.30', .5: '0.50', .7: '0.70', .9: '0.90'}, id='threshold', tooltip={'always_visible': True}))],
                          className='controls'), html.Div(id='threshold-output'), cls='controls-card'),
            note(T('Decisión de negocio'), T('Bajar el umbral detecta más clientes en riesgo pero dispara más falsas alertas. Elegirlo bien requiere conocer el costo de cada contacto y de cada cliente perdido.'))]
