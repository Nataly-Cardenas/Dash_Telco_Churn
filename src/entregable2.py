"""Resultados de los entregables 2 y 3 (112 configuraciones): se conservan como pestaña de la sección Modelos.

Corresponden al capítulo MODELADO_AVANZADO del Jupyter Book (con los complementos del entregable 3). Usan predicciones y tablas guardadas; no cargan modelos.
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
from src.plots import BASE, DIVERGING, GREY, HOT, SEQ, SOFT, bar_labels, go, highlight, make_subplots, px, roc_animation
from src.ui import (BOOK, card, cat, chart, config_name, counter, details, dropdown, eyebrow, field, graph, grid, kpi, kpis, label,
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
    resultados.append(robustez())
    calibracion = [calibration_summary()]
    optimizadores = [card(field(T('Familia'), dropdown('opt-family', list(a.MODELS), 'random_forest', fam)), cls='controls-card'), html.Div(id='opt-output'), optimizer_effects()]
    return html.Div([note(T('Entregable 2'), T('Siete familias × cuatro balanceos × cuatro optimizadores = 112 configuraciones evaluadas con validación cruzada anidada y F1 como métrica. Se conservan como referencia; el modelo propio se comparó después con 11 benchmarks recalibrados.')),
                     html.Div([podium(), kpis(kpi(T('CONFIGURACIONES'), 112, '7 × 4 × 4'), kpi(T('UNIDADES EXTERNAS'), 1680, T('112 × 3 semillas × 5 folds')),
                                              kpi('FOLDS', '5 × 3', T('externos × internos')), kpi(T('PRESUPUESTO'), 24, T('evaluaciones por búsqueda')), cls='k2')], className='hero'),
                     tabs('model-tabs', [(T('1 · Resultados por modelo'), resultados), (T('2 · Calibración'), calibracion), (T('3 · Optimizadores'), optimizadores),
                                         (T('4 · Multifidelidad'), multifidelidad()), (T('5 · Eficiencia computacional'), eficiencia()),
                                         (T('6 · Comparación estadística'), estadistica() + effect_sizes()), (T('7 · Interpretabilidad'), interpretabilidad()), (T('8 · Umbral de alerta'), umbral_lab()), (T('9 · Conclusiones'), [conclusiones()])])])


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




REPRO = [('file', N('Versiones exactas'), N('requirements.txt fija cada librería usada en el entrenamiento.')),
         ('shield', N('Verificación automática'), N('verificar_reproduccion.py compara versiones, huellas SHA-256 de los datos y los artefactos de las 1,680 unidades.')),
         ('layers', N('Resultados versionados'), N('Pipelines, predicciones, checkpoints y semillas de cada unidad quedan guardados; el libro se construye sin reentrenar.')),
         ('code', N('Código del experimento'), N('Un apéndice reproduce las 32 celdas exactas que entrenaron las 112 configuraciones.'))]


def conclusiones():
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
    return html.Div([subhead(T('Hallazgos principales'), T('Conclusiones de los entregables 2 y 3; las del modelo propio están en el capítulo 9 del libro.')),
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


@callback(Output('models-output', 'children'), Input('model-family', 'value'), Input('model-balance', 'value'), Input('model-metric', 'value'), State('lang', 'data'))
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
                     grid(chart(T('Diversidad del algoritmo genético'), fd, T('La población pierde diversidad con las generaciones: converge hacia pocas configuraciones.'), 340),
                          chart(T('Presupuesto y coste empírico'), fb, T('Segundos por unidad externa en el equipo del experimento.'), 340, False))])


@callback(Output('cal-map-output', 'children'), Input('cal-family', 'value'), State('lang', 'data'))
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


@callback(Output('eff-variants-output', 'children'), Input('eff-family', 'value'), State('lang', 'data'))
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


@callback(Output('eff-scale-output', 'children'), Input('eff-scale', 'value'), State('lang', 'data'))
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
