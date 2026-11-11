"""Carga de artefactos, validación de entradas, predicción y explicación SHAP local.

Los modelos se cargan una sola vez por proceso (`lru_cache`). El dashboard nunca entrena: si falta un artefacto,
la sección de predicción muestra un aviso y el resto de la aplicación sigue funcionando.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline_telco import BENCHMARKS, ENTRADA, ORIGINAL

RAIZ = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("ARTIFACTS_DIR", RAIZ / "artifacts"))
MODELOS = [ORIGINAL] + BENCHMARKS
NOMBRES = {"RegionalMAPEM": "RegionalMAPEM (modelo propio)", "knn": "KNN", "naive_bayes": "Naive Bayes",
           "logistic": "Regresión logística", "decision_tree": "Árbol de decisión", "random_forest": "Random Forest",
           "xgboost": "XGBoost", "svm": "SVM RBF", "logistic_splines": "Logística con splines", "hist_gb": "HistGradientBoosting",
           "lightgbm": "LightGBM", "catboost": "CatBoost"}
SHAP_FONDO, SHAP_COALICIONES = 12, 100          # explicación en vivo: 12 clientes de referencia y 100 coaliciones


class EntradaInvalida(ValueError):
    """Error de validación de un cliente: el mensaje se muestra tal cual en la interfaz."""


@lru_cache(maxsize=None)
def metadata(modelo):
    return json.loads((ART / modelo / "metadata.json").read_text(encoding="utf-8"))


def disponibles():
    return [m for m in MODELOS if (ART / m / "modelo.joblib").exists()]


@lru_cache(maxsize=1)
def cargar():
    """Los 12 artefactos, cargados una vez. RegionalMAPEM se enlaza con los 11 benchmarks ya cargados."""
    modelos = {m: joblib.load(ART / m / "modelo.joblib") for m in BENCHMARKS}
    modelos[ORIGINAL] = joblib.load(ART / ORIGINAL / "modelo.joblib").conectar(modelos)
    return modelos


def esquema():
    return {c["nombre"]: c for c in metadata(ORIGINAL)["esquema_entrada"]}


def validar(valores):
    """Convierte el formulario en un DataFrame de una fila; lanza EntradaInvalida con un mensaje claro."""
    esq = esquema()
    fila, errores = {}, []
    for c in ENTRADA:
        e, v = esq[c], valores.get(c)
        if v is None or (isinstance(v, str) and not v.strip()):
            errores.append(f"Falta {c}.")
            continue
        if e["tipo"] == "numérica":
            try:
                v = float(v)
            except (TypeError, ValueError):
                errores.append(f"{c} debe ser un número.")
                continue
            if not np.isfinite(v) or v < e["min"] or v > e["max"]:
                errores.append(f"{c} debe estar entre {e['min']:g} y {e['max']:g}.")
                continue
            fila[c] = int(v) if c in ("tenure", "SeniorCitizen") else v
        else:
            if v not in e["categorias"]:
                errores.append(f"{c} no admite el valor {v!r}.")
                continue
            fila[c] = v
    if not errores:
        if fila["tenure"] == 0 and fila["TotalCharges"] > fila["MonthlyCharges"]:
            errores.append("Con tenure = 0 el cargo acumulado no puede superar un mes de cargo.")
        if fila["tenure"] > 0 and fila["TotalCharges"] < 1:
            errores.append("Con tenure > 0 el cargo acumulado debe ser positivo.")
        if fila["PhoneService"] == "No" and fila["MultipleLines"] != "No phone service":
            errores.append("Sin servicio telefónico, MultipleLines debe ser 'No phone service'.")
        if fila["PhoneService"] == "Yes" and fila["MultipleLines"] == "No phone service":
            errores.append("Con servicio telefónico, MultipleLines no puede ser 'No phone service'.")
        sin_internet = fila["InternetService"] == "No"
        for c in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"):
            if sin_internet != (fila[c] == "No internet service"):
                errores.append(f"{c} debe ser 'No internet service' si y solo si InternetService = 'No'.")
                break
    if errores:
        raise EntradaInvalida(" ".join(errores))
    return pd.DataFrame([fila], columns=ENTRADA)


def predecir(modelo, X):
    return cargar()[modelo].predict_proba(X)[:, 1]


@lru_cache(maxsize=1)
def fondo_shap():
    """Clientes de referencia de SHAP: muestra fija del conjunto de desarrollo (TotalCharges ya numérico)."""
    raw = pd.read_csv(RAIZ / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    ids = pd.read_csv(RAIZ / "data" / "train_index.csv")["train_index"].to_numpy()
    X = raw.loc[ids, ENTRADA].reset_index(drop=True)
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
    rng = np.random.default_rng(20261110)
    return X.iloc[np.sort(rng.choice(len(X), SHAP_FONDO, replace=False))].reset_index(drop=True)


def funcion_modelo(modelo, columnas, tipos):
    """f(A) para SHAP: arma un DataFrame con los tipos originales y devuelve la probabilidad de abandono."""
    def f(A):
        d = pd.DataFrame(A, columns=columnas)
        for c in columnas:
            d[c] = d[c].astype(tipos[c])
        return predecir(modelo, d)
    return f


def shap_local(modelo, X, fondo=None, coaliciones=SHAP_COALICIONES, semilla=0):
    """Valores SHAP (KernelExplainer, enlace identidad) de un cliente. Suman p(x) − E[p(fondo)]."""
    import shap
    fondo = fondo_shap() if fondo is None else fondo
    X = X.copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
    tipos = fondo.dtypes.to_dict()
    np.random.seed(semilla)
    ex = shap.KernelExplainer(funcion_modelo(modelo, list(fondo.columns), tipos), fondo.to_numpy(dtype=object), link="identity")
    valores = np.asarray(ex.shap_values(X[fondo.columns].to_numpy(dtype=object), nsamples=coaliciones, silent=True))
    return pd.Series(valores.reshape(-1), index=fondo.columns), float(np.ravel(ex.expected_value)[0])
