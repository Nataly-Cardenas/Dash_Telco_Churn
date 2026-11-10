"""Pruebas del tablero: datos, tres páginas, callbacks, traducciones, paleta, asistente y endpoints."""
import ast
import json
import re
from pathlib import Path

import numpy as np
from plotly.utils import PlotlyJSONEncoder

import app
import pages.contexto as contexto
import pages.eda as eda_page
import pages.modelos as modelos
from src import data as a
from src import entregable2 as e2
from src import modelos as M

RAIZ = Path(__file__).resolve().parents[1]
FILTROS = [('train', 'Todos', 'Todos', [0, 72]), ('train', 'Two year', 'Fiber optic', [0, 0]), ('test', 'Month-to-month', 'DSL', [0, 12])]


def js(x): return json.dumps(x, cls=PlotlyJSONEncoder, ensure_ascii=False)


def test_original_split_and_counts():
    assert len(a.RAW) == 7043
    assert len(a.TRAIN_IDS) == 5634 and len(a.TEST_IDS) == 1409
    assert not set(a.TRAIN_IDS) & set(a.TEST_IDS)
    assert set(a.TRAIN_IDS) | set(a.TEST_IDS) == set(a.RAW.index)
    assert a.RAW.target.sum() == 1869
    assert a.RAW.TotalCharges.isna().sum() == 11
    assert a.RAW.loc[a.RAW.TotalCharges.isna(), 'tenure'].eq(0).all()


def test_tres_paginas_registradas():
    import dash
    rutas = {p['path'] for p in dash.page_registry.values()}
    assert rutas == {'/', '/eda', '/modelos'}
    assert [p for p, _, _ in app.NAV] == ['/', '/eda', '/modelos']
    client = app.server.test_client()
    for url in ['/', '/eda', '/modelos', '/healthz', '/_dash-layout', '/_dash-dependencies', '/artifacts/RegionalMAPEM/metadata.json',
                '/evidence/random_forest__smote__bayesian_tpe/lime_local_4386.png', '/assets/img/laura.jpg']:
        assert client.get(url).status_code == 200, url
    assert client.get('/healthz').get_json()['models'] == 12
    assert client.get('/artifacts/no_existe/metadata.json').status_code == 404
    assert client.get('/evidence/../../app.py').status_code == 404


def test_predicciones_guardadas_coinciden_con_el_experimento():
    P = a.predicciones_finales()
    v, h = P[P.evaluacion == 'validacion_r0'], P[P.evaluacion == 'holdout']
    assert len(v) == 5634 and set(v.row_id) == set(a.TRAIN_IDS)
    assert len(h) == 1409 and set(h.row_id) == set(a.TEST_IDS)
    assert np.array_equal(P.y.to_numpy(), a.RAW.loc[P.row_id, 'target'].to_numpy())
    from sklearn.metrics import log_loss
    c = a.resultado('corridas_finales.csv').query("corrida == 'holdout'").set_index('modelo')
    for m in M.MODELOS:
        assert abs(log_loss(h.y, h[m]) - c.loc[m, 'log_loss']) < 1e-5, m


def test_contexto_renderiza_y_cita_la_fuente():
    for lang in ['es', 'en']:
        page = js(contexto.render(lang, 1))
        assert 'hbr.org' in page and 'Kaggle' in page
    customers = contexto.cover_customers()
    assert len(customers) == 30 and len({c['id'] for c in customers}) == 30
    assert all(0 <= c['prob'] <= 1 for c in customers)


def test_eda_callbacks_serializan():
    for f in FILTROS:
        js(eda_page.render('es', *f))
        for var in a.NUMERIC:
            js(eda_page.uni_num(var, *f)); js(eda_page.bi_num(var, *f))
        for var in ['__all__'] + a.CATEGORICAL: js(eda_page.uni_cat(var, *f))
        for var in a.CATEGORICAL: js(eda_page.bi_cat(var, *f))
        js(eda_page.multi(*f)); js(eda_page.missing(*f))
        for x, y, c in [('tenure', 'MonthlyCharges', 'Churn'), ('Contract', 'TotalCharges', 'PaymentMethod'), ('tenure', 'tenure', 'Churn')]:
            js(eda_page.explorador_xy(x, y, c, *f))


def test_filtros_y_cohortes():
    d = a.cohort('train', 'Month-to-month', 'Fiber optic', [0, 12])
    assert len(d) > 0 and d.tenure.le(12).all()
    assert d.Contract.eq('Month-to-month').all() and d.InternetService.eq('Fiber optic').all()
    assert a.cohort('train', 'Two year', 'Fiber optic', [0, 0]).empty
    assert eda_page.reset(1) == ('train', 'Todos', 'Todos', [0, 72])


def test_eda_statistics_match_the_book():
    from src import eda
    d = a.RAW.loc[a.TRAIN_IDS]
    num, cat, fam = eda.holm_family(d)
    assert len(fam) == 19
    assert round(num.set_index('Variable').loc['tenure', 'r punto-biserial'], 4) == -0.3456
    v = cat.set_index('Variable')['V de Cramér']
    assert v.idxmax() == 'Contract' and round(v.max(), 4) == 0.4111
    assert (eda.normality(d)['¿Normal?'] == 'No').all()


def test_modelos_render_y_tabla_de_lideres():
    for lang in ['es', 'en']:
        js(modelos.render(lang, 1))
    d = modelos.lideres()
    principales = d[d.tipo != 'ablacion']
    assert len(principales) == 12 and set(principales.id) == set(M.MODELOS)
    assert principales.iloc[0].id in {'RegionalMAPEM', 'logistic_splines'}
    assert (d.lo <= d.ll).all() and (d.ll <= d.hi).all()
    tabla = modelos.tabla_lideres()
    assert tabla.sort_action == 'native' and tabla.filter_action == 'native' and len(tabla.data) == 12
    assert len(modelos.tabla_lideres(True).data) == 17


def test_prediccion_en_vivo_y_errores_claros():
    vals = modelos.cargar_cliente(int(a.TEST_IDS[3]))[:-1]
    out = js(modelos.predecir(1, 'logistic_splines', None, *vals, 'es'))
    assert 'SHAP' in out and 'conforme' in out.lower()
    malo = list(vals); malo[M.ENTRADA.index('tenure')] = 500
    assert 'tenure debe estar entre 0 y 72' in js(modelos.predecir(1, 'knn', None, *malo, 'es'))
    vacio = list(vals); vacio[M.ENTRADA.index('Contract')] = None
    assert 'Falta Contract' in js(modelos.predecir(1, 'knn', None, *vacio, 'es'))
    assert modelos.cliente_azar(3) in set(a.TEST_IDS)


def test_conforme_cubre_cerca_del_nominal():
    for m in M.MODELOS:
        q, cobertura, tamano = modelos.conforme(m)
        assert 0 < q < 1 and 0.85 < cobertura < 0.95 and 1 <= tamano <= 2


def test_diagnostico_e_interpretabilidad():
    for m in M.MODELOS:
        for ev in ['validacion_r0', 'holdout']: js(modelos.diagnostico_salida(m, ev))
        js(modelos.shap_salida(m))
    s = modelos.shap_largo()
    assert set(s.modelo) == set(M.MODELOS) and s.groupby('modelo').row_id.nunique().eq(150).all()


def test_entregable2_se_conserva():
    rf = a.SUMMARY.set_index('experiment_id').loc[a.FINALISTS[0]]
    assert len(a.SUMMARY) == 112 and np.isclose(rf.f1_positive_mean, .6384205660936855)
    for model in a.MODELS:
        js(e2.model_output(model, 'Todos', 'f1_positive')); js(e2.opt_output(model)); js(e2.res_output(model))
    for model in a.FINALISTS[:2]: js(e2.explain_output(model))
    for model in a.FINALISTS: js(e2.threshold_output(model, .3))


def test_download_only_on_click():
    from dash import no_update
    assert eda_page.export(0, ['customerID'], 'train', 'Todos', 'Todos', [0, 72]) is no_update
    payload = eda_page.export(1, ['customerID', 'Churn'], 'train', 'Todos', 'Todos', [0, 72])
    assert payload['filename'] == 'telco_seleccion.csv' and len(payload['content'].splitlines()) == 5635


def archivos_con_texto():
    return [RAIZ / 'app.py', *sorted((RAIZ / 'pages').glob('*.py')), *[RAIZ / 'src' / n for n in ['chatbot.py', 'entregable2.py', 'ui.py', 'plots.py']]]


def test_text_files_always_specify_utf8():
    for ruta in [RAIZ / 'src' / 'data.py', *archivos_con_texto(), RAIZ / 'src' / 'modelos.py']:
        for node in ast.walk(ast.parse(ruta.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'read_text':
                assert any(k.arg == 'encoding' and isinstance(k.value, ast.Constant) and k.value.value == 'utf-8' for k in node.keywords), ruta.name


def test_every_visible_text_has_english_translation():
    from src.i18n import EN
    texts = []
    for ruta in archivos_con_texto():
        for node in ast.walk(ast.parse(ruta.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('T', 'N') and node.args and isinstance(node.args[0], ast.Constant):
                texts.append(node.args[0].value)
    for d in [a.LABELS, a.MODELS, a.BALANCES, a.OPTIMIZERS, a.FINALIST_NAMES]: texts += list(d.values())
    missing = sorted({t for t in texts if t not in EN})
    assert not missing, missing


def test_pages_render_in_english_without_emoji():
    from src.i18n import set_lang
    emoji = re.compile('[\U0001F000-\U0001FAFF☀-➿️]')
    try:
        for lang in ['en', 'es']:
            for page in [contexto.render(lang, 1), eda_page.render(lang, *FILTROS[0]), modelos.render(lang, 1)]:
                assert not emoji.search(js(page))
            assert ('Lost' in js(app.chrome(lang))) == (lang == 'en')
        assert 'Leaderboard' in js(modelos.render('en', 1))
    finally:
        set_lang('es')


def test_charts_use_one_color_plus_highlight():
    from src.plots import BASE, GREY, HOT
    allowed = {BASE.lower(), HOT.lower(), GREY.lower()}
    vals = modelos.cargar_cliente(int(a.TEST_IDS[3]))[:-1]
    payload = js([eda_page.render('es', *FILTROS[0]), modelos.render('es', 1), modelos.diagnostico_salida('RegionalMAPEM', 'holdout'),
                  modelos.shap_salida('RegionalMAPEM'), modelos.predecir(1, 'RegionalMAPEM', None, *vals, 'es'),
                  eda_page.bi_cat('Contract', *FILTROS[0]), eda_page.explorador_xy('tenure', 'MonthlyCharges', 'PaymentMethod', *FILTROS[0])])

    def drop_templates(x):
        if isinstance(x, dict): return {k: drop_templates(v) for k, v in x.items() if k != 'template'}
        if isinstance(x, list): return [drop_templates(v) for v in x]
        return x
    used = {c.lower() for c in re.findall(r'"(#[0-9A-Fa-f]{6})"', json.dumps(drop_templates(json.loads(payload))))}
    assert used <= allowed | {'#0b1426', '#eaf2ff', '#8792a8'}, used - allowed


def test_chatbot_answers_from_project_knowledge():
    from src import chatbot
    es = chatbot.local_answer('¿cuál fue el mejor modelo?', 'es')
    assert 'RegionalMAPEM' in es and 'Random Forest' in es and '0.638' in es
    en = chatbot.local_answer('which customers churn the most?', 'en')
    assert 'electronic check' in en.lower()
    assert 'CSV' in chatbot.local_answer('cómo descargo el csv', 'es')
    assert 'Predicción en vivo' in chatbot.local_answer('cómo hago una predicción', 'es')
    t = a.RAW.loc[a.TRAIN_IDS]; expected = t[(t.Contract == 'Month-to-month') & (t.InternetService == 'Fiber optic')].target.mean()
    assert f'{expected:.1%}' in chatbot.data_answer('tasa de abandono de clientes con fibra y mes a mes', 'es')
    assert chatbot.data_answer('quién hizo el proyecto', 'es') is None


def test_el_dashboard_no_entrena():
    """Ningún módulo del servidor llama a fit: solo se cargan artefactos."""
    for ruta in [RAIZ / 'app.py', *sorted((RAIZ / 'pages').glob('*.py')), RAIZ / 'src' / 'modelos.py', RAIZ / 'src' / 'entregable2.py']:
        src = ruta.read_text(encoding='utf-8')
        assert '.fit(' not in src and 'fit_from_predictions' not in src, ruta.name
