"""Pipelines de Telco del experimento confirmatorio, para reentrenar y cargar los artefactos.

Las clases y funciones de este módulo son copia literal de `experiments/confirmatorio/comun.py` del repositorio
Telco_Churn_Entregable (preprocesamiento, splines y construcción de los 11 benchmarks calibrados con Platt). Se copian
aquí para que los artefactos serializados dependan de un módulo del propio dashboard y no del repositorio del
experimento. `scripts/construir_artefactos.py` comprueba que los pipelines reentrenados con este módulo reproducen
las predicciones guardadas por el experimento.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

EPS = 1e-6
BENCHMARKS = ["knn", "naive_bayes", "logistic", "decision_tree", "random_forest", "xgboost", "svm", "logistic_splines",
              "hist_gb", "lightgbm", "catboost"]
ORIGINAL = "RegionalMAPEM"
SEED_ROLES = {"subset": 11, "outer": 23, "inner": 37, "sampler": 41, "model": 53, "calibration": 67,
              "test": 79, "optimizer": 83}
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "AvgChargePerMonth"]
CATEGORICAS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
# columnas de entrada de los pipelines, en el orden del archivo original (sin customerID ni Churn)
ENTRADA = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
           "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
           "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]


def seed_for(role, *coords):
    """Misma derivación de semillas que el experimento (raíz 42)."""
    return int(np.random.SeedSequence([42, SEED_ROLES[role], *coords]).generate_state(1)[0])


class HistoricalFeatures(TransformerMixin, BaseEstimator):
    """ETL por filas de Telco (no aprende nada): TotalCharges, servicios y AvgChargePerMonth."""
    SERVICE = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        total = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if (total.isna() & out["tenure"].ne(0)).any():
            raise ValueError("TotalCharges ausente con tenure != 0")
        out["TotalCharges"] = total.fillna(0.0)
        for c in self.SERVICE:
            out[c] = out[c].replace("No internet service", "No")
        out["MultipleLines"] = out["MultipleLines"].replace("No phone service", "No")
        avg = out["MonthlyCharges"].to_numpy(float).copy()
        np.divide(out["TotalCharges"].to_numpy(float), out["tenure"].to_numpy(float), out=avg,
                  where=out["tenure"].to_numpy() > 0)
        out["AvgChargePerMonth"] = avg
        return out


def preprocesador():
    """Preprocesamiento de Telco. Todo se ajusta dentro del conjunto de entrenamiento."""
    return Pipeline([("features", HistoricalFeatures()), ("prep", ColumnTransformer(
        [("num", StandardScaler(), NUMERICAS),
         ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAS)],
        remainder="drop", sparse_threshold=0.0))])


class SplinesContinuas(TransformerMixin, BaseEstimator):
    """B-splines cúbicos solo para las columnas continuas (más de 2 valores en el ajuste)."""

    def __init__(self, n_knots=5, knots="uniform", degree=3):
        self.n_knots = n_knots
        self.knots = knots
        self.degree = degree

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.transformadores_ = {}
        for j in range(X.shape[1]):
            x = X[:, j]
            if len(np.unique(x)) <= 2:
                continue
            q = np.linspace(0, 100, self.n_knots)
            grid = np.unique(np.percentile(x, q)) if self.knots == "quantile" else np.linspace(x.min(), x.max(), self.n_knots)
            if len(grid) < 2:
                continue
            self.transformadores_[j] = SplineTransformer(knots=grid[:, None], degree=self.degree,
                                                         extrapolation="constant").fit(x[:, None])
        self.pasan_ = [j for j in range(X.shape[1]) if j not in self.transformadores_]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        partes = [t.transform(X[:, [j]]) for j, t in self.transformadores_.items()]
        return np.hstack(partes + [X[:, self.pasan_]])


def estimador(nombre, p, semilla, fijos=None):
    """Estimador sin calibrar con los hiperparámetros `p` más los fijos del espacio de búsqueda."""
    p = {**(fijos or {}), **p}
    if nombre == "knn":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"])
    if nombre == "naive_bayes":
        return GaussianNB(var_smoothing=p["var_smoothing"])
    if nombre == "logistic":
        return LogisticRegression(solver=p["solver"], max_iter=p["max_iter"], C=p["C"], penalty=p["penalty"],
                                  random_state=semilla)
    if nombre == "decision_tree":
        return DecisionTreeClassifier(random_state=semilla, **{k: p[k] for k in ("max_depth", "min_samples_leaf", "criterion")})
    if nombre == "random_forest":
        return RandomForestClassifier(random_state=semilla, n_jobs=1, **p)
    if nombre == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=semilla, n_jobs=1, verbosity=0, **p)
    if nombre == "svm":
        return SVC(kernel=p["kernel"], C=p["C"], gamma=p["gamma"], cache_size=500, random_state=semilla)
    if nombre == "logistic_splines":
        return Pipeline([("spl", SplinesContinuas(p["n_knots"], p["knots"], p["degree"])),
                         ("lr", LogisticRegression(C=p["C"], solver=p["solver"], penalty=p["penalty"],
                                                   max_iter=p["max_iter"], random_state=semilla))])
    if nombre == "hist_gb":
        return HistGradientBoostingClassifier(random_state=semilla, **p)
    if nombre == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=semilla, n_jobs=1, **p)
    if nombre == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=semilla, allow_writing_files=False, **p)
    raise ValueError(nombre)


def pipeline_calibrado(nombre, p, semilla, fijos=None):
    """Preprocesamiento + modelo, calibrado con Platt (cv=3)."""
    return CalibratedClassifierCV(Pipeline([("pre", preprocesador()), ("modelo", estimador(nombre, p, semilla, fijos))]),
                                  method="sigmoid", cv=3)


class EnsambleRegional:
    """RegionalMAPEM listo para predecir desde las 19 variables originales de un cliente.

    Guarda el preprocesamiento (para las regiones) y el combinador ajustado. Los 11 expertos no se serializan dentro:
    son los artefactos de los benchmarks y se enlazan al cargar con `conectar`, para no tenerlos dos veces en memoria.
    """

    def __init__(self, preprocesamiento, combinador, expertos):
        self.preprocesamiento = preprocesamiento
        self.combinador = combinador
        self.expertos = list(expertos)          # nombres, en el orden de las columnas del ajuste
        self.modelos_ = None

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado["modelos_"] = None
        return estado

    def conectar(self, modelos):
        faltan = [m for m in self.expertos if m not in modelos]
        if faltan:
            raise ValueError(f"Faltan expertos para RegionalMAPEM: {faltan}")
        self.modelos_ = [modelos[m] for m in self.expertos]
        return self

    def probabilidades_expertos(self, X):
        if self.modelos_ is None:
            raise ValueError("Use conectar() con los 11 benchmarks antes de predecir.")
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.modelos_])

    def predict_proba(self, X, P=None):
        P = self.probabilidades_expertos(X) if P is None else P
        Z = np.asarray(self.preprocesamiento.transform(X), float)
        return self.combinador.predict_proba_from_predictions(Z, P)

    def regiones(self, X):
        """Pertenencia suave de cada cliente a las K regiones."""
        return self.combinador._regions(np.asarray(self.preprocesamiento.transform(X), float))
