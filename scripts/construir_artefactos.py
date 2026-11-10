"""Reentrena y serializa los 12 modelos finales de Telco en artifacts/ (se corre una vez, fuera del servidor).

Hiperparámetros: los que eligió la búsqueda del protocolo (30 evaluaciones de TPE sobre 3 pliegues internos) en la
corrida final con los 5634 clientes de desarrollo (`crudos/telco_holdout/r9_f0/`). Es la misma búsqueda de cada pliegue
externo de la validación anidada, aplicada al conjunto de desarrollo completo.

Para cada modelo:
  1. reentrena el pipeline calibrado sobre los 5634 clientes con su semilla del protocolo;
  2. comprueba que sus probabilidades sobre el hold-out coinciden con las guardadas por el experimento;
  3. lo guarda con joblib (compress=3), lo vuelve a cargar y verifica que predice exactamente lo mismo;
  4. escribe artifacts/<modelo>/metadata.json (versiones, semilla, hiperparámetros, esquema y métricas externas).

Uso: python scripts/construir_artefactos.py <ruta del repositorio Telco_Churn_Entregable>
"""
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.original_model import RegionalMAPEM  # noqa: E402
from src.pipeline_telco import (BENCHMARKS, ENTRADA, ORIGINAL, EnsambleRegional, pipeline_calibrado,  # noqa: E402
                                preprocesador, seed_for)

ART = RAIZ / "artifacts"
COORD = (9, 0)                       # coordenadas de semilla de la corrida final del protocolo
LIBRERIAS = ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "xgboost", "lightgbm", "catboost"]


def datos():
    raw = pd.read_csv(RAIZ / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    raw.columns = raw.columns.str.strip()

    def parte(archivo, col):
        ids = pd.read_csv(RAIZ / "data" / archivo)[col].astype(int).to_numpy()
        return raw.loc[ids].drop(columns=["customerID", "Churn"]).reset_index(drop=True), \
            raw.loc[ids, "Churn"].map({"No": 0, "Yes": 1}).astype(int).to_numpy(), ids
    return parte("train_index.csv", "train_index"), parte("test_index.csv", "test_index")


def esquema(X):
    """Nombres, tipos y rangos válidos de las 19 variables de entrada (del conjunto de desarrollo)."""
    out = []
    for c in ENTRADA:
        s = X[c]
        if c == "TotalCharges":
            v = pd.to_numeric(s, errors="coerce")
            out.append({"nombre": c, "tipo": "numérica", "dtype": "float (texto en el archivo original)", "min": 0.0,
                        "max": float(v.max()), "nota": "vacío solo si tenure = 0; se imputa con 0"})
        elif pd.api.types.is_numeric_dtype(s):
            out.append({"nombre": c, "tipo": "numérica", "dtype": str(s.dtype), "min": float(s.min()), "max": float(s.max())})
        else:
            out.append({"nombre": c, "tipo": "categórica", "dtype": "str", "categorias": sorted(s.unique().tolist())})
    return out


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def entrenar(nombre, X, y, params, fijos):
    semilla = seed_for("model", *COORD, BENCHMARKS.index(nombre))
    t0 = time.perf_counter()
    modelo = pipeline_calibrado(nombre, params, semilla, fijos).fit(X, y)
    return nombre, modelo, semilla, time.perf_counter() - t0


def main(proyecto):
    proyecto = Path(proyecto)
    crudos = proyecto / "experiments" / "resultados" / "crudos" / "telco_holdout" / f"r{COORD[0]}_f{COORD[1]}"
    tablas = proyecto / "experiments" / "resultados" / "tablas"
    espacios = json.loads((proyecto / "experiments" / "espacios.json").read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "-C", str(proyecto), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (X, y, _), (Xh, yh, ids_h) = datos()
    assert len(X) == 5634 and len(Xh) == 1409

    exp = {m: pickle.load(open(crudos / f"{m}.pkl", "rb")) for m in BENCHMARKS}
    com = pickle.load(open(crudos / "_combinadores.pkl", "rb"))[ORIGINAL]
    resumen = pd.read_csv(tablas / "resumen_modelos.csv").query("conjunto == 'telco'").set_index("modelo")
    hold = pd.read_csv(tablas / "libro" / "corridas_finales.csv").query("corrida == 'holdout'").set_index("modelo")
    costos = pd.read_csv(tablas / "costos_por_pliegue.csv").query("conjunto == 'telco'").groupby("modelo")

    entrenados = Parallel(n_jobs=11)(delayed(entrenar)(m, X, y, exp[m]["params"], espacios["benchmarks"][m].get("fijos", {}))
                                     for m in BENCHMARKS)
    modelos = {m: mod for m, mod, _, _ in entrenados}
    semillas = {m: s for m, _, s, _ in entrenados}
    segundos = {m: t for m, _, _, t in entrenados}

    # RegionalMAPEM: regiones y pesos con las predicciones fuera de pliegue de la búsqueda final de cada experto
    O = np.column_stack([exp[m]["oof"] for m in BENCHMARKS])
    pp = preprocesador().fit(X)
    Z = np.asarray(pp.transform(X), float)
    fijos = espacios["modelo_original"]["RegionalMAPEM"]["fijos"]
    bp = com["params"]
    semilla_o = seed_for("model", *COORD, 100)
    t0 = time.perf_counter()
    comb = RegionalMAPEM(n_regions=bp["K"], prior_strength=bp["c"], covariance=bp["covarianza"],
                         gmm_n_init=fijos["n_init_gmm"], max_iter=fijos["max_iter_em"], tol=fijos["tol_relativa_em"],
                         random_state=semilla_o).fit_from_predictions(Z, y, O, expert_names=BENCHMARKS)
    segundos[ORIGINAL] = time.perf_counter() - t0
    ens = EnsambleRegional(pp, comb, BENCHMARKS).conectar(modelos)

    versiones = {"python": platform.python_version(), **{lib: version(lib) for lib in LIBRERIAS}}
    ahora = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    esq = esquema(X)
    indice = []
    for m in BENCHMARKS + [ORIGINAL]:
        obj = ens if m == ORIGINAL else modelos[m]
        guardado = (com if m == ORIGINAL else exp[m])["pte"]
        p = obj.predict_proba(Xh)[:, 1]
        dif_exp = float(np.max(np.abs(p - guardado)))
        destino = ART / m
        destino.mkdir(parents=True, exist_ok=True)
        ruta = destino / "modelo.joblib"
        joblib.dump(obj, ruta, compress=3)
        cargado = joblib.load(ruta)
        if m == ORIGINAL:
            cargado.conectar({b: joblib.load(ART / b / "modelo.joblib") for b in BENCHMARKS})
        p2 = cargado.predict_proba(Xh)[:, 1]
        assert np.array_equal(p, p2), f"ida y vuelta distinta en {m}"
        r, h = resumen.loc[m], hold.loc[m]
        t1 = time.perf_counter()
        obj.predict_proba(Xh.iloc[:200])
        ms_cliente = 1000 * (time.perf_counter() - t1) / 200
        meta = {
            "modelo": m,
            "tipo": "modelo original" if m == ORIGINAL else "benchmark",
            "creado": ahora,
            "fuente": {"repositorio": "https://github.com/Nataly-Cardenas/Telco_Churn_Entregable", "commit": commit,
                       "corrida": f"experiments/resultados/crudos/telco_holdout/r{COORD[0]}_f{COORD[1]}/"},
            "versiones": versiones,
            "semilla": {"raiz": 42, "derivacion": f"seed_for('model', {COORD[0]}, {COORD[1]}, "
                                                  f"{100 if m == ORIGINAL else BENCHMARKS.index(m)})",
                        "valor": semilla_o if m == ORIGINAL else semillas[m]},
            "entrenamiento": {"clientes": int(len(X)), "abandonos": int(y.sum()), "segundos_ajuste": round(segundos[m], 3),
                              "calibracion": "expertos calibrados con Platt (cv = 3); combinación sin recalibrar" if m == ORIGINAL
                              else "CalibratedClassifierCV(method='sigmoid', cv=3)", "balanceo": "ninguno"},
            "hiperparametros": ({**bp, "n_init_gmm": fijos["n_init_gmm"], "max_iter_em": fijos["max_iter_em"],
                                 "tol_relativa_em": fijos["tol_relativa_em"], "expertos": BENCHMARKS} if m == ORIGINAL
                                else {**espacios["benchmarks"][m].get("fijos", {}), **exp[m]["params"]}),
            "busqueda": {"metodo": "TPE (optuna)", "evaluaciones": int((com if m == ORIGINAL else exp[m])["ensayos"]),
                         "objetivo": "log-loss medio de 3 pliegues internos",
                         "log_loss_interno": float(min(t["valor"] for t in (com if m == ORIGINAL else exp[m])["registro"]))},
            "umbral_f1": float((com if m == ORIGINAL else exp[m])["umbral"]),
            "esquema_entrada": esq,
            "salida": "probabilidad de abandono en [0, 1] (predict_proba[:, 1])",
            "metricas_externas": {
                "validacion_anidada": {"diseno": "5 repeticiones × 5 pliegues externos (25 evaluaciones)",
                                       **{f"{k}_{s}": float(r[f"{k}_{s}"]) for k in ("log_loss", "f1", "auc", "brier", "ece")
                                          for s in ("media", "de")},
                                       "log_loss_ic95": [float(r.log_loss_ic_bajo), float(r.log_loss_ic_alto)]},
                "holdout": {"clientes": int(h.n), **{k: float(h[k]) for k in ("log_loss", "f1", "auc", "brier", "ece", "umbral")}},
            },
            "costo": {"ms_por_cliente_lote_200": round(ms_cliente, 3),
                      "seg_pred_por_1000_validacion": None if m == ORIGINAL else float(costos.get_group(m).seg_pred_por_1000.mean()),
                      "tamano_mb": round(ruta.stat().st_size / 1e6, 3)},
            "verificacion": {"max_dif_vs_experimento_holdout": dif_exp, "ida_y_vuelta_identica": True,
                             "sha256": sha256(ruta)},
        }
        (destino / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        indice.append({"modelo": m, "tamano_mb": meta["costo"]["tamano_mb"], "max_dif_vs_experimento": dif_exp})
        print(f"{m:18s} {meta['costo']['tamano_mb']:7.3f} MB  dif. vs experimento {dif_exp:.2e}", flush=True)

    # conjunto de prueba fijo para la prueba de ida y vuelta de tests/
    muestra = np.random.default_rng(seed_for("test", 500)).choice(len(Xh), 40, replace=False)
    fijo = Xh.iloc[np.sort(muestra)].copy()
    fijo.insert(0, "row_id", ids_h[np.sort(muestra)])
    for m in BENCHMARKS + [ORIGINAL]:
        fijo[f"p_{m}"] = (ens if m == ORIGINAL else modelos[m]).predict_proba(fijo[ENTRADA])[:, 1]
    fijo.to_csv(ART / "prueba_ida_y_vuelta.csv", index=False, float_format="%.17g")
    (ART / "indice.json").write_text(json.dumps({"creado": ahora, "commit_experimento": commit, "modelos": indice},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RAIZ.parent.parent / "Desktop" / "Telco_Churn_Entregable")
