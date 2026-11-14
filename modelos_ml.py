"""Sección «Modelos finales» (entregable de ML) integrada al dash original:  tabla de líderes, predicción en vivo con SHAP, diagnóstico, comparación estadística,
modelo original y ficha del modelo. Solo carga artefactos ya entrenados (artifacts/)."""
import time
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, log_loss, roc_auc_score

from src import data as a
from src import modelos as M
from src.i18n import N, T, set_lang
from src.icons import icon
from src.plots import (BASE, GREY, HOT, SEQ, bars_ci, cd_diagram, confusion, gauge, go, highlight, make_subplots, precision_recall, px,
                       reliability, roc, shap_beeswarm, shap_waterfall)
from src.ui import (alert, card, cat, chart, config_name, details, dropdown, eyebrow, field, graph, grid, kpi, kpis, label, md, note,
                    page_head, subhead, table, tabs, tt)


BEST_BY_FAMILY = {'knn': 'knn__smote__bayesian_tpe', 'naive_bayes': 'naive_bayes__none__random', 'logistic': 'logistic__smote__grid', 'decision_tree': 'decision_tree__smote__bayesian_tpe',
                  'random_forest': 'random_forest__smote__bayesian_tpe', 'xgboost': 'xgboost__smote__random', 'svm': 'svm__smote__genetic'}
ORIGINAL, MEJOR = 'RegionalMAPEM', 'logistic_splines'
NOMBRES = {**{k: N(v) for k, v in M.NOMBRES.items()}, 'RegionalMAPEM': N('RegionalMAPEM (modelo propio)'),
           'mapem_sin_prior': N('RegionalMAPEM sin prior'), 'mapem_K1': N('Pesos globales (K = 1)'), 'mapem_regiones_duras': N('Regiones duras'),
           'promedio_simple': N('Promedio simple'), 'stacking_logistico': N('Stacking logístico')}
ABLACION = ['mapem_sin_prior', 'mapem_K1', 'mapem_regiones_duras', 'promedio_simple', 'stacking_logistico']
CORTO = {'RegionalMAPEM': 'RegionalMAPEM', 'logistic_splines': 'Splines', 'hist_gb': 'HistGB', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost',
         'xgboost': 'XGBoost', 'random_forest': 'RF', 'logistic': N('Logística'), 'knn': 'KNN', 'naive_bayes': 'NB', 'decision_tree': N('Árbol'), 'svm': 'SVM',
         'mapem_sin_prior': N('Sin prior'), 'mapem_K1': 'K = 1', 'mapem_regiones_duras': N('Duras'), 'promedio_simple': N('Promedio'), 'stacking_logistico': 'Stacking'}
ALFA = .10


def nombre(m): return T(NOMBRES.get(m, m))
def corto(m): return T(CORTO.get(m, m))
def pm(m, s, d=4): return f'{m:.{d}f} ± {s:.{d}f}'


# ---------------------------------------------------------------- contenido de la sección
def contenido(lang):
    """Devuelve la sección completa; la ruta del dash la vuelve a pintar al cambiar el idioma."""
    set_lang(lang)
    v = a.resultado('veredicto_R3.json')
    return html.Div([
        page_head(T('Modelos finales'), T('Doce modelos finales reentrenados con los 5634 clientes de desarrollo: once benchmarks calibrados y RegionalMAPEM, el modelo propio.'), '06_validacion'),
        kpis(kpi(T('LOG-LOSS DEL MODELO PROPIO'), round(v['log_loss_original'], 5), T('validación anidada 5 × 5'), 'hot', 5),
             kpi(T('MEJOR BENCHMARK'), round(v['log_loss_mejor_benchmark'], 5), T('logística con splines'), '', 5),
             kpi(T('DIFERENCIA'), f"{v['diferencia']:+.5f}", T('no significativa con ninguna prueba')),
             kpi(T('VEREDICTO R3'), T('No superior'), T('empate con el mejor benchmark'))),
        tabs('modelos-tabs', [(T('Tabla de líderes'), leaderboard()), (T('Predicción en vivo'), prediccion()), (T('Diagnóstico'), diagnostico()),
                              (T('Interpretabilidad global'), interpretabilidad()), (T('Comparación estadística'), comparacion()),
                              (T('Modelo original'), original()), (T('Ficha del modelo'), ficha())])])


# ---------------------------------------------------------------- tabla de líderes
@lru_cache(maxsize=1)
def lideres():
    r = a.telco('resumen_modelos.csv').set_index('modelo')
    costos = a.resultado('costos_telco.csv').groupby('modelo').seg_pred_por_1000.mean()
    filas = []
    for m in r.index:
        x = r.loc[m]
        meta = M.metadata(m) if m in M.MODELOS else None
        filas.append({'id': m, 'tipo': 'original' if m == ORIGINAL else 'ablacion' if m in ABLACION else 'benchmark',
                      'll': x.log_loss_media, 'll_de': x.log_loss_de, 'lo': x.log_loss_ic_bajo, 'hi': x.log_loss_ic_alto,
                      'f1': x.f1_media, 'f1_de': x.f1_de, 'auc': x.auc_media, 'auc_de': x.auc_de, 'brier': x.brier_media, 'ece': x.ece_media,
                      'hold': meta['metricas_externas']['holdout']['log_loss'] if meta else np.nan,
                      'ms': meta['costo']['ms_por_cliente_lote_200'] if meta else np.nan,
                      'ms_val': costos.get(m, np.nan), 'mb': meta['costo']['tamano_mb'] if meta else np.nan})
    return pd.DataFrame(filas).sort_values('ll').reset_index(drop=True)


def tabla_lideres(incluir_ablacion=False):
    d = lideres()
    if not incluir_ablacion: d = d[d.tipo != 'ablacion']
    d = d.assign(pos=range(1, len(d) + 1))
    t = pd.DataFrame({'#': d.pos, T('Modelo'): d.id.map(nombre), T('Tipo'): d.tipo.map({'original': T('Modelo propio'), 'benchmark': 'Benchmark', 'ablacion': T('Ablación o combinador')}),
                      T('Balanceo'): T('Ninguno'), T('Optimizador'): 'TPE · 30', T('Calibración'): 'Platt',
                      T('Log-loss (media ± DE)'): [pm(m, s) for m, s in zip(d.ll, d.ll_de)], T('Log-loss'): d.ll.round(5),
                      T('IC 95 % log-loss'): [f'[{lo:.4f}, {hi:.4f}]' for lo, hi in zip(d.lo, d.hi)],
                      T('F1 (media ± DE)'): [pm(m, s, 3) for m, s in zip(d.f1, d.f1_de)], 'F1': d.f1.round(4),
                      T('AUC (media ± DE)'): [pm(m, s, 3) for m, s in zip(d.auc, d.auc_de)], 'AUC': d.auc.round(4),
                      'Brier': d.brier.round(4), 'ECE': d.ece.round(4), T('Log-loss hold-out'): d.hold.round(4),
                      T('ms por cliente'): d.ms.round(3), T('Tamaño (MB)'): d.mb.round(3)})
    return table(t, id='tabla-lideres', page_size=17, filter_action='native', decimals=5,
                 style_data_conditional=[{'if': {'filter_query': '{%s} = "%s"' % (T('Modelo'), nombre(ORIGINAL))}, 'color': HOT, 'fontWeight': '600'}],
                 tooltip_header={T('IC 95 % log-loss'): T('Intervalo t sobre las 25 evaluaciones externas'), T('ms por cliente'): T('Predicción de un lote de 200 clientes, en milisegundos por cliente')})


def leaderboard():
    d = lideres(); d = d[d.tipo != 'ablacion']
    f = bars_ci([nombre(m) for m in d.id], d.ll, d.lo, d.hi, nombre(ORIGINAL), T('Log-loss medio externo (menor es mejor) · IC 95 %'))
    e2 = a.SUMMARY.set_index('experiment_id').loc[list(BEST_BY_FAMILY.values())].reset_index().sort_values('f1_positive_mean', ascending=False)
    rangos = pd.read_csv(a.DATA / 'estadistica' / 'rangos_medios.csv').set_index('experiment_id').mean_rank
    import json
    cd = json.loads((a.DATA / 'estadistica' / 'nemenyi_estado.json').read_text(encoding='utf-8'))['critical_difference']
    mejor = rangos.min()
    tb = pd.DataFrame({T('Familia'): e2.model.map(lambda m: T(a.MODELS[m])), T('Balanceo'): e2.balance.map(lambda b: T(a.BALANCES[b])), T('Optimizador'): e2.optimizer.map(lambda o: T(a.OPTIMIZERS[o])),
                       T('F1 (media ± DE)'): [pm(m, s, 3) for m, s in zip(e2.f1_positive_mean, e2.f1_positive_std)],
                       T('AUC (media ± DE)'): [pm(m, s, 3) for m, s in zip(e2.roc_auc_mean, e2.roc_auc_std)],
                       T('Rango medio Friedman'): e2.experiment_id.map(rangos).round(1),
                       T('Finalista Nemenyi'): [T('Sí') if rangos[e] - mejor <= cd else 'No' for e in e2.experiment_id],
                       T('Motivo'): [T('Mejor combinación global') if i == 0 else T('Mayor F1 medio externo de su familia') for i in range(len(e2))]})
    return [note(T('Cómo se eligieron'), T('Cada benchmark se optimizó con TPE (30 evaluaciones por pliegue externo) minimizando el log-loss del pipeline calibrado con Platt, sin remuestreo. '
                                         'La tabla resume las 25 evaluaciones externas (5 repeticiones × 5 pliegues); el modelo propio está resaltado. Ordena o filtra cualquier columna.')),
            card(dcc.Checklist([{'label': ' ' + T('Incluir la ablación y los combinadores estándar'), 'value': 'si'}], [], id='lideres-ablacion', className='check'),
                 html.Div(tabla_lideres(), id='lideres-tabla')),
            chart(T('Log-loss externo con su intervalo de confianza'), f, T('Los intervalos de los cinco primeros se solapan casi por completo: RegionalMAPEM (resaltado) y la logística con splines empatan.'), 480, False),
            subhead(T('Mejor combinación de cada modelo base del entregable 2'), T('Selección del bloque B sobre las 112 configuraciones (7 modelos × 4 balanceos × 4 optimizadores), por F1 medio externo.')),
            card(table(tb, page_size=7), html.P(T('Finalista Nemenyi: su rango medio está a menos de una diferencia crítica ({cd}) del mejor. Las siete mejores combinaciones quedan dentro del grupo de no rechazo.').format(cd=f'{cd:.1f}'), className='insight'))]


@callback(Output('lideres-tabla', 'children'), Input('lideres-ablacion', 'value'), State('lang', 'data'), prevent_initial_call=True)
def cambiar_lideres(v, lang='es'):
    set_lang(lang)
    return tabla_lideres(bool(v))


# ---------------------------------------------------------------- predicción en vivo
@lru_cache(maxsize=1)
def holdout_raw():
    return a.RAW.loc[a.TEST_IDS]


@lru_cache(maxsize=None)
def conforme(modelo):
    """Umbral conforme al 90 % con las predicciones externas de la validación (5634) y su cobertura en el hold-out."""
    P = a.predicciones_finales()
    v, h = P[P.evaluacion == 'validacion_r0'], P[P.evaluacion == 'holdout']
    s = np.where(v.y == 1, 1 - v[modelo], v[modelo])
    n = len(s); q = float(np.quantile(s, min(1, np.ceil((n + 1) * (1 - ALFA)) / n), method='higher'))
    incluye1, incluye0 = 1 - h[modelo] <= q, h[modelo] <= q
    cubre = np.where(h.y == 1, incluye1, incluye0)
    return q, float(cubre.mean()), float((incluye1.astype(int) + incluye0.astype(int)).mean())


def control(c):
    e = M.esquema()[c]
    if e['tipo'] == 'categórica':
        return dcc.Dropdown([{'label': cat(v), 'value': v} for v in e['categorias']], None, id='in-' + c, clearable=False)
    if c == 'SeniorCitizen':
        return dcc.Dropdown([{'label': cat(0), 'value': 0}, {'label': cat(1), 'value': 1}], None, id='in-' + c, clearable=False)
    paso = 1 if c == 'tenure' else .01
    return dcc.Input(id='in-' + c, type='number', min=e['min'], max=e['max'], step=paso, debounce=True, className='num-input',
                     placeholder=f"{e['min']:g} – {e['max']:g}")


GRUPOS_FORM = [(N('Cuenta'), ['tenure', 'Contract', 'PaymentMethod', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges']),
               (N('Servicios'), ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']),
               (N('Demografía'), ['gender', 'SeniorCitizen', 'Partner', 'Dependents'])]


def prediccion():
    if len(M.disponibles()) < len(M.MODELOS):
        return [alert(T('Faltan artefactos'), T('No se encontraron todos los modelos en artifacts/. Ejecuta scripts/construir_artefactos.py; el tablero no entrena modelos.'), 'danger')]
    h = holdout_raw()
    ejemplos = [{'label': f'{cid} · {T("abandonó") if t else T("permaneció")}', 'value': int(i)} for i, cid, t in zip(h.index, h.customerID, h.target)]
    form = html.Div([html.Div([html.H4(T(g)), dbc.Row([dbc.Col(field(label(c) + f' · {c}', control(c)), xs=12, md=6, xl=4) for c in cs], className='g-3')], className='form-group') for g, cs in GRUPOS_FORM])
    return [html.Div([
        card(html.Div([field(T('Modelo'), dropdown('pred-modelo', M.MODELOS, ORIGINAL, {m: nombre(m) for m in M.MODELOS})),
                       field(T('Cargar un cliente del hold-out'), dcc.Dropdown(ejemplos, int(a.TEST_IDS[3]), id='pred-ejemplo', clearable=False)),
                       html.Button([icon('shuffle'), T('Cliente al azar')], id='pred-azar', className='btn ghost', n_clicks=0)], className='controls'),
             html.P(T('El formulario se genera con el esquema de entrada guardado en metadata.json: tipos, categorías válidas y rangos observados en el desarrollo. Puedes editar cualquier campo.'), className='muted'),
             form,
             html.Div([html.Button([icon('target'), T('Predecir')], id='pred-boton', className='btn primary', n_clicks=0),
                       html.Small(T('La explicación SHAP tarda unos segundos para RegionalMAPEM: combina a 11 modelos.'), className='muted')], className='export-row'),
             cls='controls-card'),
        dcc.Store(id='pred-cargado'), dcc.Loading(html.Div(id='pred-salida'), type='dot', color=BASE)], className='pred-layout'),
        note(T('Qué muestra la predicción'), T('Probabilidad calibrada de abandono, la decisión con el umbral que maximiza el F1 en la validación, el conjunto conforme al 90 % y la explicación SHAP local. '
                                               'Es una herramienta de apoyo: no reemplaza la revisión humana antes de actuar sobre un cliente.'))]


@callback(*[Output('in-' + c, 'value') for c in M.ENTRADA], Output('pred-cargado', 'data'), Input('pred-ejemplo', 'value'))
def cargar_cliente(row):
    """Llena el formulario con un cliente real; la predicción se recalcula después (pred-cargado)."""
    c = a.RAW.loc[row]
    return [int(c[k]) if k in ('tenure', 'SeniorCitizen') else (None if pd.isna(c[k]) else float(c[k])) if k in ('MonthlyCharges', 'TotalCharges') else c[k] for k in M.ENTRADA] + [int(row)]


@callback(Output('pred-ejemplo', 'value'), Input('pred-azar', 'n_clicks'), prevent_initial_call=True)
def cliente_azar(n): return int(np.random.default_rng(n).choice(a.TEST_IDS))


@lru_cache(maxsize=256)
def explicar(modelo, fila):
    X = pd.DataFrame([dict(fila)], columns=M.ENTRADA)
    t0 = time.perf_counter()
    v, base = M.shap_local(modelo, X)
    return v, base, time.perf_counter() - t0


@callback(Output('pred-salida', 'children'), Input('pred-boton', 'n_clicks'), Input('pred-modelo', 'value'), Input('pred-cargado', 'data'), *[State('in-' + c, 'value') for c in M.ENTRADA], State('lang', 'data'))
def predecir(_, modelo, _cargado, *args):
    *valores, lang = args
    set_lang(lang)
    entrada = dict(zip(M.ENTRADA, valores))
    if entrada['tenure'] == 0 and entrada.get('TotalCharges') is None: entrada['TotalCharges'] = 0.0
    try:
        X = M.validar(entrada)
    except M.EntradaInvalida as e:
        return alert(T('Revisa los datos del cliente'), str(e), 'warning')
    try:
        p = float(M.predecir(modelo, X)[0])
        v, base, seg = explicar(modelo, tuple(X.iloc[0].items()))
    except Exception as e:                       # un error del modelo no debe detener la aplicación
        return alert(T('No se pudo calcular la predicción'), f'{type(e).__name__}: {e}', 'danger')
    meta = M.metadata(modelo)
    u = meta['umbral_f1']; q, cob, tam = conforme(modelo)
    conjunto = [T('Abandona')] * (1 - p <= q) + [T('Permanece')] * (p <= q)
    nivel = HOT if p >= u else BASE
    fw = shap_waterfall(v, base, lambda c: f'{label(c)} = {cat(X.iloc[0][c]) if isinstance(X.iloc[0][c], str) else X.iloc[0][c]}')
    out = [kpis(kpi(T('PROBABILIDAD DE ABANDONO'), round(p * 100, 1), nombre(modelo), 'hot' if p >= u else '', 1, ' %'),
                kpi(T('DECISIÓN'), T('Alerta de retención') if p >= u else T('Sin alerta'), T('umbral F1 = {u}').format(u=f'{u:.3f}')),
                kpi(T('CONJUNTO CONFORME 90 %'), ' + '.join(conjunto) if conjunto else '∅', T('cobertura en hold-out: {c}').format(c=f'{cob:.1%}')),
                kpi(T('SHAP CALCULADO EN'), f'{seg:.1f} s', T('{n} clientes de referencia').format(n=M.SHAP_FONDO))),
           grid(card(html.H3(T('Probabilidad predicha'), className='chart-title'), graph(gauge(p, nombre(modelo), nivel), 260, False, {'l': 34, 'r': 34, 't': 50, 'b': 10}),
                     html.P(T('La aguja es la probabilidad calibrada. Con el umbral que maximiza el F1 en validación ({u}), este cliente {d}.').format(u=f'{u:.2f}', d=T('recibe una alerta') if p >= u else T('no recibe alerta')), className='insight')),
                card(html.H3(T('Incertidumbre: predicción conforme'), className='chart-title'),
                     html.P(T('Con cobertura nominal del 90 %, el conjunto de predicción contiene las clases cuya puntuación de no conformidad no supera q = {q}. '
                              'En el hold-out, estos conjuntos contienen la clase real en el {c} de los clientes, con {t} clases en promedio.').format(q=f'{q:.3f}', c=f'{cob:.1%}', t=f'{tam:.2f}')),
                     html.P(T('Un conjunto con las dos clases significa que el modelo no tiene certeza suficiente sobre este cliente.') if len(conjunto) == 2 else
                            T('Un conjunto con una sola clase indica una predicción con la confianza requerida.'), className='insight'))),
           chart(T('Explicación SHAP local'), fw, T('Cada barra es lo que una variable mueve la probabilidad desde la media de referencia ({b}) hasta la de este cliente ({p}). Resaltado: empuja hacia el abandono.').format(b=f'{base:.3f}', p=f'{p:.3f}'), 460, False)]
    if modelo == ORIGINAL:
        ens = M.cargar()[ORIGINAL]
        R = ens.regiones(X)[0]; W = R @ ens.combinador.weights_
        s = pd.Series(W, index=M.BENCHMARKS).sort_values()
        fp = go.Figure(go.Bar(x=s.values, y=[corto(m) for m in s.index], orientation='h', marker_color=highlight(s.values), text=[f'{x:.2f}' for x in s.values], textposition='outside'))
        fp.update_xaxes(title=T('Peso efectivo del experto para este cliente'), range=[0, max(.5, s.max() * 1.25)])
        out.append(chart(T('Pesos de los expertos para este cliente'), fp, T('Pertenencia a las {k} regiones: {r}. Los pesos efectivos son la mezcla de los pesos de cada región; el experto con más peso está resaltado.').format(
            k=len(R), r=', '.join(f'{x:.0%}' for x in R)), 360, False))
    return html.Div(out)


# ---------------------------------------------------------------- diagnóstico
def diagnostico():
    return [card(html.Div([field(T('Modelo'), dropdown('diag-modelo', M.MODELOS, ORIGINAL, {m: nombre(m) for m in M.MODELOS})),
                           field(T('Evaluación'), dcc.RadioItems([{'label': ' ' + T('Validación anidada (repetición 0, 5634 clientes)'), 'value': 'validacion_r0'},
                                                                   {'label': ' ' + T('Hold-out (1409 clientes, una sola vez)'), 'value': 'holdout'}], 'validacion_r0', id='diag-eval', className='radio'))],
                          className='controls'), cls='controls-card'),
            dcc.Loading(html.Div(id='diag-salida'), type='dot', color=BASE)]


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.mean(idx == b) * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if np.any(idx == b)))


@callback(Output('diag-salida', 'children'), Input('diag-modelo', 'value'), Input('diag-eval', 'value'), State('lang', 'data'))
def diagnostico_salida(modelo, evaluacion, lang='es'):
    set_lang(lang)
    P = a.predicciones_finales(); d = P[P.evaluacion == evaluacion]
    U = a.resultado('umbrales.csv')
    u = d.pliegue.map(lambda f: U.loc[f'{evaluacion}_{f}', f'u_{modelo}']).to_numpy()
    y, p = d.y.to_numpy(), np.clip(d[modelo].to_numpy(), 1e-6, 1 - 1e-6)
    pred = (p >= u).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fcm = px.imshow(cm, x=[T('Predice permanencia'), T('Predice abandono')], y=[T('Permanece'), T('Abandona')], text_auto=True, color_continuous_scale=SEQ, aspect='auto')
    fcm.update_coloraxes(showscale=False); fcm.update_traces(textfont_size=17)
    fuente = T('5 pliegues externos de la repetición 0; cada pliegue usa su propio umbral') if evaluacion == 'validacion_r0' else T('hold-out, modelo final entrenado con los 5634 clientes')
    tn, fp_, fn, tp = cm.ravel()
    return html.Div([eyebrow(nombre(modelo) + ' · ' + fuente),
                     kpis(kpi('LOG-LOSS', round(log_loss(y, p), 4), T('menor es mejor'), 'hot', 4), kpi('F1', round(f1_score(y, pred), 3), T('umbral que maximiza F1 en validación'), '', 3),
                          kpi('AUC', round(roc_auc_score(y, p), 3), '', '', 3), kpi('BRIER', round(brier_score_loss(y, p), 4), T('ECE = {e}').format(e=f'{ece(y, p):.3f}'), '', 4)),
                     grid(chart(T('Matriz de confusión'), fcm, T('Detecta {tp} de {n} abandonos con {fp} falsas alertas.').format(tp=tp, n=tp + fn, fp=fp_), 330, False),
                          chart(T('Diagrama de calibración'), reliability(y, p), T('Más cerca de la diagonal = probabilidades mejor calibradas. El tamaño del punto indica cuántos clientes hay en el intervalo.'), 330)),
                     grid(chart(T('Curva ROC'), roc(y, p, f'AUC = {roc_auc_score(y, p):.3f}'), T('Capacidad de ordenar a los clientes por riesgo.'), 330),
                          chart(T('Curva precisión-recall'), precision_recall(y, p, nombre(modelo)), T('La línea gris es la prevalencia del abandono: lo que lograría un modelo sin información.'), 330))])


# ---------------------------------------------------------------- interpretabilidad global
@lru_cache(maxsize=1)
def shap_largo():
    S = a.resultado('shap_global.csv')
    largo = S.melt(id_vars=['modelo', 'row_id', 'base'], value_vars=M.ENTRADA, var_name='variable', value_name='shap')
    X = a.RAW.loc[S.row_id.unique(), M.ENTRADA].copy(); X['TotalCharges'] = X.TotalCharges.fillna(0)
    tasas = a.RAW.loc[a.TRAIN_IDS]
    color = {}
    for c in M.ENTRADA:
        if c in a.NUMERIC:
            color[c] = X[c].rank(pct=True)
        else:
            t = tasas.groupby(c).target.mean(); t = (t - t.min()) / max(t.max() - t.min(), 1e-12)
            color[c] = X[c].map(t)
    C = pd.DataFrame(color)
    largo['color'] = [C.at[r, v] for r, v in zip(largo.row_id, largo.variable)]
    return largo


def interpretabilidad():
    return [card(field(T('Modelo'), dropdown('shap-modelo', M.MODELOS, ORIGINAL, {m: nombre(m) for m in M.MODELOS})), cls='controls-card'),
            html.Div(id='shap-salida'),
            note(T('Explicar no es causar'), T('Las atribuciones describen cómo decide el modelo, no qué provoca el abandono.'))]


@callback(Output('shap-salida', 'children'), Input('shap-modelo', 'value'), State('lang', 'data'))
def shap_salida(modelo, lang='es'):
    set_lang(lang)
    s = shap_largo(); s = s[s.modelo == modelo]
    imp = s.groupby('variable').shap.apply(lambda v: v.abs().mean()).sort_values()
    fb = go.Figure(go.Bar(x=imp.values, y=[label(c) for c in imp.index], orientation='h', marker_color=highlight(imp.values), text=[f'{v:.3f}' for v in imp.values], textposition='outside'))
    fb.update_xaxes(title=T('|SHAP| medio (cambio en la probabilidad de abandono)'))
    top = imp.index[-3:][::-1]
    return grid(chart(T('Resumen SHAP · 150 clientes del hold-out'), shap_beeswarm(s, label), T('Cada punto es un cliente. A la derecha empuja hacia el abandono. Color: valor de la variable (en las categóricas, alto = categoría con más abandono en el desarrollo).'), 560, False),
                chart(T('Importancia media'), fb, T('Las tres variables con más peso son {a}, {b} y {c}.').format(a=label(top[0]), b=label(top[1]), c=label(top[2])), 560, False))


# ---------------------------------------------------------------- comparación estadística
def comparacion():
    fr = a.resultado('friedman_telco.json')
    rangos = fr['rangos']
    f = cd_diagram(rangos, fr['CD'], ORIGINAL, {m: corto(m) for m in rangos})
    fam = a.resultado('familia_resultados.csv')
    t = fam[fam.calculada.astype(bool)].copy()
    vista = pd.DataFrame({T('Bloque'): t.bloque, T('Conjunto'): t.dataset, T('Métrica'): t.metrica, T('Prueba'): t.prueba, T('Rival'): t.modelo_b.map(lambda m: nombre(m) if isinstance(m, str) else m),
                          T('Diferencia media'): t.dif_media, 'p': t.p_crudo, T('p Holm'): t.p_holm, 'Cliff δ': t.cliff, T('Magnitud'): t.magnitud,
                          T('Pliegues que gana'): t.pliegues_gana.astype('Int64').astype(str) + '/' + t.n_pliegues.astype('Int64').astype(str)})
    rechazos = int((t.p_holm < .05).sum())
    return [kpis(kpi('FRIEDMAN F_F', round(fr['F_F'], 1), f"p = {fr['p']:.1e}", '', 1), kpi(T('MODELOS · BLOQUES'), f"{fr['k']} · {fr['N']}", T('modelos × evaluaciones externas')),
                 kpi(T('DIFERENCIA CRÍTICA'), round(fr['CD'], 2), 'Nemenyi, α = 0.05', '', 2), kpi(T('CONTRASTES CON HOLM'), len(t), T('{r} rechazan H0').format(r=rechazos))),
            chart(T('Diagrama de diferencia crítica · Telco, log-loss'), f, T('Rango medio de cada modelo en las 25 evaluaciones externas (1 = mejor). Las barras azules unen modelos que Nemenyi no distingue; '
                                                                             'RegionalMAPEM (resaltado) comparte grupo con la logística con splines, LightGBM, HistGradientBoosting y su propia ablación.'), 520, False),
            card(html.H3(T('Familia confirmatoria de contrastes'), className='chart-title'),
                 html.P(T('Las 223 comparaciones pre-registradas del modelo propio contra cada rival, con p ajustado por Holm sobre toda la familia. Filtra por bloque, prueba o rival.'), className='muted'),
                 table(vista, page_size=12, filter_action='native', decimals=5),
                 html.P(T('Contra la logística con splines en Telco: p Holm = 1 en las tres pruebas y δ de Cliff = −0,008, muy por debajo del umbral de 0,147.'), className='insight'))]


# ---------------------------------------------------------------- modelo original
PSEUDO = '''```text
Entrada: X (n × p), y, OOF[n × 11] de los expertos calibrados, K, c, covarianza
1. Z  ← cuantiles normales de las columnas continuas de X → 2 componentes principales
2. r  ← responsabilidades de un GMM con K componentes sobre Z (EM propio, 3 inicios)
3. g  ← pesos globales: EM con K = 1 y sin prior
4. α_k ← 1 + c · g                           (prior Dirichlet hacia lo global)
5. π_k ← 1/11 para cada región k
6. repetir hasta que el cambio relativo de F sea ≤ 1e-9:
       E: q_ikm ∝ r_ik · π_km · L_im          (L_im = verosimilitud del experto m)
       M: π_km ∝ Σ_i q_ikm + α_km − 1
7. predecir: p(x) = Σ_k r_k(x) Σ_m π_km p_m(x)
```'''


def original():
    v = a.resultado('veredicto_R3.json'); conv = a.resultado('convergencia.json')
    lp = np.array(conv['log_posterior'])
    fc = go.Figure(go.Scatter(x=np.linspace(0, conv['n_iter'], len(lp)), y=lp, mode='lines', line={'color': BASE, 'width': 2.4}, name='F(π)'))
    fc.update_xaxes(title=T('Iteración del MAP-EM')); fc.update_yaxes(title=T('Log-posterior F(π)'))
    busq = conv['busqueda_log_loss_interno']
    fs = go.Figure()
    for m in [ORIGINAL, MEJOR, 'hist_gb', 'lightgbm']:
        acc = np.minimum.accumulate(busq[m])
        fs.add_trace(go.Scatter(x=list(range(1, len(acc) + 1)), y=acc, mode='lines', name=nombre(m), line={'color': HOT if m == ORIGINAL else (BASE if m == MEJOR else GREY), 'width': 3 if m == ORIGINAL else 2}))
    fs.update_xaxes(title=T('Evaluaciones de TPE')); fs.update_yaxes(title=T('Mejor log-loss interno acumulado'))
    fam = a.resultado('familia_resultados.csv')
    h = fam[(fam.modelo_b == MEJOR) & fam.bloque.isin(['A1_principal_benchmarks', 'A3_principal_friedman', 'C_holdout'])]
    pruebas = pd.DataFrame({T('Prueba'): h.descripcion.fillna(h.prueba), T('Conjunto'): h.dataset, T('Métrica'): h.metrica, T('Diferencia media'): h.dif_media,
                            'p': h.p_crudo, T('p Holm'): h.p_holm, 'Cliff δ': h.cliff, T('Rechaza H0'): h.p_holm < .05})
    rob = a.resultado('robustez_resumen.csv')
    rob = rob[rob.modelo.isin([ORIGINAL, MEJOR])]
    sem = rob[rob.escenario.str.startswith('semilla')]
    fb = go.Figure([go.Box(y=sem[sem.modelo == m].log_loss, name=nombre(m), marker_color=HOT if m == ORIGINAL else BASE, boxpoints='all', jitter=.4, pointpos=0) for m in [ORIGINAL, MEJOR]])
    fb.update_yaxes(title=T('Log-loss externo (10 semillas × 5 pliegues)'))
    esc = rob[~rob.escenario.str.startswith('semilla') & (rob.escenario != 'conforme')].groupby(['escenario', 'modelo']).log_loss.mean().unstack()
    etiquetas = {'ruido_x_05': N('Ruido en X, 5 %'), 'ruido_x_10': N('Ruido en X, 10 %'), 'ruido_x_20': N('Ruido en X, 20 %'), 'ruido_y_05': N('Etiquetas cambiadas, 5 %'),
                 'ruido_y_10': N('Etiquetas cambiadas, 10 %'), 'atipicos_01': N('Atípicos, 1 %'), 'atipicos_05': N('Atípicos, 5 %'), 'n_10': N('10 % de la muestra'), 'n_25': N('25 % de la muestra'), 'n_50': N('50 % de la muestra')}
    resumen = pd.DataFrame({T('Escenario'): [T(etiquetas.get(e, e)) for e in esc.index], 'RegionalMAPEM': esc[ORIGINAL].round(4), T('Logística con splines'): esc[MEJOR].round(4),
                            T('Diferencia'): (esc[ORIGINAL] - esc[MEJOR]).round(4)})
    W = a.resultado('pesos_por_region_r0_f0.csv')
    fw = px.imshow(W.to_numpy(), x=[corto(m) for m in W.columns], y=[T('Región {k}').format(k=k + 1) for k in range(len(W))], text_auto='.2f', aspect='auto',
                   color_continuous_scale=SEQ)
    fw.update_coloraxes(showscale=False)
    condiciones = [(T('Mejor log-loss medio que el mejor benchmark'), v['condicion_1']), (T('Al menos 3 pruebas significativas tras Holm'), v['condicion_2']),
                   (T('|δ de Cliff| ≥ 0,147'), v['condicion_3']), (T('Consistencia en los 3 conjuntos adicionales'), v['condicion_4'])]
    return [
        card(html.Div([html.Div([eyebrow(T('VEREDICTO EXPLÍCITO')), html.H2(T('RegionalMAPEM no es superior al mejor benchmark')),
                                 html.P(T('Empata con la logística con splines en Telco (diferencia de {d} en log-loss, δ de Cliff = {c}) y solo es el mejor en media en Iranian. '
                                          'Cumple 1 de las 4 condiciones pre-registradas de superioridad (R3).').format(d=f"{v['diferencia']:+.5f}", c=f"{v['cliff']:+.3f}"))]),
                       html.Ul([html.Li([icon('check' if ok else 'x', 'sm'), ' ', t], className='ok' if ok else 'bad') for t, ok in condiciones], className='checks')], className='verdict'), cls='glow'),
        grid(card(html.H3(T('Descripción'), className='chart-title'),
                  md(T('RegionalMAPEM combina las probabilidades de los 11 benchmarks calibrados con pesos que cambian según la región del cliente: '
                       '$p(y=1\\mid x)=\\sum_k r_k(x)\\sum_m \\pi_{km}\\,p_m(x)$. Las regiones salen de un GMM sobre dos componentes principales de las variables continuas, '
                       'y los pesos se estiman por máxima posterior con EM, con un prior de Dirichlet $\\pi_k\\sim\\mathrm{Dir}(1+c\\,g)$ que los encoge hacia los pesos globales $g$. '
                       'Con regiones fijas la log-posterior es cóncava y el EM converge al óptimo global.')),
                  html.H3(T('Diferencias con la literatura'), className='chart-title'),
                  html.Ul([html.Li(T(x)) for x in [N('Frente a la mezcla de expertos con EM: los expertos no se reentrenan y la compuerta no es supervisada; solo se estiman los pesos.'),
                                                   N('Frente a la selección dinámica: no elige un clasificador por vecindad, mezcla a todos con pesos suaves.'),
                                                   N('Frente al stacking local: no ajusta un metamodelo por región, sino pesos en el símplex con un prior hacia lo global.'),
                                                   N('Frente a un ensamble ponderado: con K = 1 se reduce a él; el prior controla cuánto se aleja cada región.')]])),
             card(html.H3(T('Pseudocódigo'), className='chart-title'), dcc.Markdown(PSEUDO, className='prose code'),
                  html.P(T('Implementación propia en numpy y scipy (src/original_model.py), con 12 pruebas unitarias que corren en integración continua.'), className='insight'))),
        grid(chart(T('Convergencia del MAP-EM · modelo final'), fc, T('La log-posterior nunca baja y se estabiliza tras {n} iteraciones (K = {k}, c = {c}, covarianza {cov}).').format(
                 n=conv['n_iter'], k=conv['params']['K'], c=f"{conv['params']['c']:.2f}", cov=conv['params']['covarianza']), 340, False),
             chart(T('Curva de la búsqueda de hiperparámetros'), fs, T('Mejor log-loss interno acumulado en las 30 evaluaciones de la búsqueda final (5634 clientes).'), 340)),
        card(html.H3(T('Pruebas de hipótesis frente a la logística con splines'), className='chart-title'), table(pruebas, page_size=8, decimals=5, wrap=True),
             html.P(T('Ninguna prueba rechaza H0 tras Holm. El McNemar del hold-out (F1) tiene p crudo bajo, pero no sobrevive a la corrección.'), className='insight')),
        subhead(T('Resumen de los estudios de robustez'), T('Repetición 0, cinco pliegues externos; detalle completo en el capítulo 7 del libro.')),
        grid(chart(T('Estabilidad por semilla'), fb, T('La logística con splines no depende de la semilla; RegionalMAPEM cambia menos de 0,0005 entre semillas, muy por debajo de la variación entre pliegues.'), 360, False),
             card(html.H3(T('Ruido, atípicos y tamaño de muestra'), className='chart-title'), table(resumen, page_size=10),
                  html.P(T('Diferencia negativa: el modelo propio tiene menor log-loss. Ninguna diferencia pasa de 0,003 y el signo cambia entre escenarios.'), className='insight'))),
        chart(T('Pesos por región · repetición 0, pliegue 0'), fw, T('Cada fila es una región. Los pesos cambian entre regiones (HistGradientBoosting va de 0,24 a 0,54), pero el modelo con pesos globales (K = 1) logra el mismo log-loss: en Telco las regiones no aportan.'), 300, False)]


# ---------------------------------------------------------------- ficha del modelo
def ficha():
    meta = M.metadata(ORIGINAL)
    filas = pd.DataFrame([(T('Creado'), meta['creado']), (T('Commit del experimento'), meta['fuente']['commit'][:12]), (T('Semilla'), f"{meta['semilla']['valor']} · {meta['semilla']['derivacion']}"),
                          (T('Hiperparámetros'), f"K = {meta['hiperparametros']['K']}, c = {meta['hiperparametros']['c']:.4f}, {meta['hiperparametros']['covarianza']}"),
                          (T('Entrenamiento'), T('{n} clientes, {k} abandonos').format(n=meta['entrenamiento']['clientes'], k=meta['entrenamiento']['abandonos'])),
                          ('scikit-learn · numpy · pandas', f"{meta['versiones']['scikit-learn']} · {meta['versiones']['numpy']} · {meta['versiones']['pandas']}"),
                          (T('Tamaño'), f"{meta['costo']['tamano_mb']} MB + {T('11 expertos')}"), ('SHA-256', meta['verificacion']['sha256'][:16] + '…')], columns=[T('Campo'), T('Valor')])
    return [grid(card(md(a.documento('MODEL_CARD.md')), cls='model-card'),
                 html.Div([card(html.H3(T('Artefacto cargado'), className='chart-title'), table(filas, page_size=10, wrap=True),
                                html.Div([html.A([icon('download'), 'metadata.json'], href=f'/artifacts/{m}/metadata.json', target='_blank', className='chip-link') for m in [ORIGINAL, MEJOR]], className='author-links')),
                           note(T('Usos no recomendados'), T('Decidir de forma automática sobre un cliente sin revisión humana, explicar causas del abandono o aplicarlo a otra población sin volver a validarlo.'), 'amber')]))]
