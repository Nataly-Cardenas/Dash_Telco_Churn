"""Artefactos de artifacts/: ida y vuelta, metadata completa, validación de entradas y SHAP local."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import modelos as M
from src.pipeline_telco import BENCHMARKS, ENTRADA, ORIGINAL

FIJO = pd.read_csv(M.ART / 'prueba_ida_y_vuelta.csv', float_precision='round_trip')   # lectura exacta de los decimales


def test_estan_los_12_artefactos_y_son_livianos():
    assert M.disponibles() == M.MODELOS
    total = sum((M.ART / m / 'modelo.joblib').stat().st_size for m in M.MODELOS) / 1e6
    assert total < 50                                    # cabe en el plan gratuito de Render y en GitHub


@pytest.mark.parametrize('modelo', BENCHMARKS)
def test_ida_y_vuelta_identica(modelo, tmp_path):
    """Serializar con joblib (compress=3) y volver a cargar da exactamente las mismas predicciones que el modelo en memoria."""
    en_memoria = joblib.load(M.ART / modelo / 'modelo.joblib')
    joblib.dump(en_memoria, tmp_path / 'm.joblib', compress=3)
    recargado = joblib.load(tmp_path / 'm.joblib')
    X = FIJO[ENTRADA]
    assert np.array_equal(en_memoria.predict_proba(X), recargado.predict_proba(X))


def test_ida_y_vuelta_identica_regionalmapem(tmp_path):
    modelos = {m: joblib.load(M.ART / m / 'modelo.joblib') for m in BENCHMARKS}
    ens = joblib.load(M.ART / ORIGINAL / 'modelo.joblib').conectar(modelos)
    joblib.dump(ens, tmp_path / 'r.joblib', compress=3)
    recargado = joblib.load(tmp_path / 'r.joblib')
    assert recargado.modelos_ is None                      # los expertos no se serializan dos veces
    recargado.conectar(modelos)
    assert np.array_equal(ens.predict_proba(FIJO[ENTRADA]), recargado.predict_proba(FIJO[ENTRADA]))


@pytest.mark.parametrize('modelo', M.MODELOS)
def test_predice_lo_mismo_que_al_construir_el_artefacto(modelo):
    """Coincide con las probabilidades guardadas al construir el artefacto (Windows). Entre sistemas operativos pueden
    cambiar los últimos bits por la BLAS y la librería matemática; en el mismo equipo la coincidencia es exacta."""
    p = M.predecir(modelo, FIJO[ENTRADA])
    assert np.allclose(p, FIJO[f'p_{modelo}'].to_numpy(), rtol=0, atol=1e-9)


@pytest.mark.parametrize('modelo', M.MODELOS)
def test_metadata_completa(modelo):
    meta = M.metadata(modelo)
    for clave in ['versiones', 'semilla', 'hiperparametros', 'esquema_entrada', 'metricas_externas', 'umbral_f1']:
        assert clave in meta
    assert meta['versiones']['scikit-learn'] == '1.6.1'
    assert meta['entrenamiento']['clientes'] == 5634
    assert {c['nombre'] for c in meta['esquema_entrada']} == set(ENTRADA)
    va, ho = meta['metricas_externas']['validacion_anidada'], meta['metricas_externas']['holdout']
    assert 0.3 < va['log_loss_media'] < 0.5 and ho['clientes'] == 1409
    assert meta['verificacion']['max_dif_vs_experimento_holdout'] == 0.0
    assert hashlib.sha256((M.ART / modelo / 'modelo.joblib').read_bytes()).hexdigest() == meta['verificacion']['sha256']


def test_metadata_coincide_con_el_experimento():
    r = pd.read_csv(M.RAIZ / 'data' / 'resultados' / 'resumen_modelos.csv').query("conjunto == 'telco'").set_index('modelo')
    for m in M.MODELOS:
        assert np.isclose(M.metadata(m)['metricas_externas']['validacion_anidada']['log_loss_media'], r.loc[m, 'log_loss_media'])
    v = json.loads((M.RAIZ / 'data' / 'resultados' / 'veredicto_R3.json').read_text(encoding='utf-8'))
    assert v['R3'] is False and np.isclose(r.loc[ORIGINAL, 'log_loss_media'], v['log_loss_original'])


def cliente():
    return dict(FIJO.iloc[0][ENTRADA])


def test_validacion_rechaza_entradas_invalidas():
    X = M.validar(cliente())
    assert list(X.columns) == ENTRADA and len(X) == 1
    for cambio, mensaje in [({'tenure': 500}, 'tenure'), ({'tenure': 'abc'}, 'número'), ({'Contract': 'Tres años'}, 'Contract'),
                            ({'MonthlyCharges': None}, 'Falta'), ({'PhoneService': 'No', 'MultipleLines': 'Yes'}, 'MultipleLines')]:
        with pytest.raises(M.EntradaInvalida, match=mensaje):
            M.validar({**cliente(), **cambio})


def test_cliente_nuevo_con_tenure_cero():
    c = {**cliente(), 'tenure': 0, 'TotalCharges': 0.0}
    c['MonthlyCharges'] = 50.0
    p = M.predecir(ORIGINAL, M.validar(c))
    assert 0 < p[0] < 1


@pytest.mark.parametrize('modelo', [ORIGINAL, 'logistic_splines'])
def test_shap_local_suma_la_prediccion(modelo):
    X = M.validar(cliente())
    v, base = M.shap_local(modelo, X)
    assert list(v.index) == ENTRADA
    assert np.isclose(v.sum() + base, M.predecir(modelo, X)[0], atol=1e-6)
