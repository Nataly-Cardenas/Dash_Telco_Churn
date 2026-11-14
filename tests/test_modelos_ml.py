"""La sección «Modelos finales» (ML) convive con el dash original sin pisar sus callbacks ni sus ids."""
import json

from plotly.utils import PlotlyJSONEncoder

import analytics as a
import app
import modelos_ml
from i18n import set_lang


def test_la_pestana_ml_se_pinta_en_ambos_idiomas():
    d = a.cohort('train', 'Todos', 'Todos', [0, 72])
    for lang in ('es', 'en'):
        set_lang(lang)
        assert json.dumps(app.PAGES['modelos_ml'](d), cls=PlotlyJSONEncoder)
    set_lang('es')


def test_las_secciones_originales_siguen_ahi():
    claves = {k for k, _, _ in app.NAV}
    assert {'introduccion', 'eda', 'preprocesamiento', 'modelado', 'simulador', 'conclusiones', 'autoras'} <= claves
    assert 'modelos_ml' in claves


def js(x): return json.dumps(x, cls=PlotlyJSONEncoder)


def test_prediccion_en_vivo_y_validaciones():
    vals = modelos_ml.cargar_cliente(int(a.TEST_IDS[3]))[:-1]
    assert js(modelos_ml.predecir(1, 'logistic_splines', None, *vals, 'es'))
    assert modelos_ml.cliente_azar(3) in set(a.TEST_IDS)


def test_diagnostico_y_shap_de_los_doce_modelos():
    for m in modelos_ml.M.MODELOS:
        for ev in ['validacion_r0', 'holdout']: js(modelos_ml.diagnostico_salida(m, ev))
        js(modelos_ml.shap_salida(m))


def test_no_hay_ids_de_componentes_repetidos_en_la_pestana_ml():
    d = a.cohort('train', 'Todos', 'Todos', [0, 72])
    ids = []

    def recorre(x):
        if isinstance(x, dict):
            if isinstance(x.get('props'), dict) and isinstance(x['props'].get('id'), str): ids.append(x['props']['id'])
            for v in x.values(): recorre(v)
        elif isinstance(x, list):
            for v in x: recorre(v)
    recorre(json.loads(json.dumps(modelos_ml.contenido('es'), cls=PlotlyJSONEncoder)))
    repetidos = {i for i in ids if ids.count(i) > 1}
    assert not repetidos, repetidos
