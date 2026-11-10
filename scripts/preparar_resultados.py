"""Copia al dashboard los resultados del experimento confirmatorio y prepara las tablas livianas de la sección Modelos.

No entrena nada: lee las tablas de `experiments/resultados/tablas/`, las predicciones guardadas en `crudos/` y la ficha
del modelo. Al final calcula el resumen SHAP global con los artefactos ya guardados en artifacts/.

Uso: python scripts/preparar_resultados.py <ruta del repositorio Telco_Churn_Entregable> [n_procesos]
"""
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.pipeline_telco import BENCHMARKS, ENTRADA, ORIGINAL, seed_for  # noqa: E402

SAL = RAIZ / "data" / "resultados"
MODELOS = BENCHMARKS + [ORIGINAL]
TABLAS = ["resumen_modelos.csv", "friedman_telco.json", "familia_resultados.csv", "veredicto_R3.json",
          "metricas_por_pliegue.csv", "memoria_por_modelo.csv", "pesos_por_region_r0_f0.csv", "shap_importancia.csv",
          "tabla_maestra_experimentos.csv", "bayesiana_rope.csv", "escalabilidad.csv",
          "libro/corridas_finales.csv", "libro/robustez_resumen.csv", "libro/conforme.csv", "libro/calibracion_telco.csv",
          "libro/errores_subgrupos.csv", "libro/conjuntos.csv"]


def copiar(proyecto):
    tablas = proyecto / "experiments" / "resultados" / "tablas"
    SAL.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for t in TABLAS:
        destino = SAL / Path(t).name
        shutil.copyfile(tablas / t, destino)
        hashes[Path(t).name] = hashlib.sha256(destino.read_bytes()).hexdigest()
    costos = pd.read_csv(tablas / "costos_por_pliegue.csv").query("conjunto == 'telco'")
    costos.drop(columns=["params"]).to_csv(SAL / "costos_telco.csv", index=False)
    for doc in ["MODEL_CARD.md", "DATA_STATEMENT.md"]:
        shutil.copyfile(proyecto / doc, RAIZ / "data" / doc)
        hashes[doc] = hashlib.sha256((RAIZ / "data" / doc).read_bytes()).hexdigest()
    commit = subprocess.run(["git", "-C", str(proyecto), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sucio = subprocess.run(["git", "-C", str(proyecto), "status", "--porcelain", "experiments/resultados/tablas"],
                           capture_output=True, text=True).stdout.strip()
    return {"repositorio": "https://github.com/Nataly-Cardenas/Telco_Churn_Entregable", "commit": commit,
            "cambios_sin_commit_en_tablas": bool(sucio), "copiado": date.today().isoformat(), "sha256": hashes}


def predicciones(proyecto):
    """Predicciones externas de la repetición 0 (los 5634 de desarrollo, 5 pliegues) y del hold-out (1409)."""
    crudos = proyecto / "experiments" / "resultados" / "crudos"
    ids_tr = pd.read_csv(RAIZ / "data" / "train_index.csv")["train_index"].to_numpy()
    ids_te = pd.read_csv(RAIZ / "data" / "test_index.csv")["test_index"].to_numpy()
    raw = pd.read_csv(RAIZ / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    y = (raw.Churn == "Yes").astype(int).to_numpy()
    filas = []
    folds = StratifiedKFold(5, shuffle=True, random_state=seed_for("outer", 0)).split(np.zeros(len(ids_tr)), y[ids_tr])
    for f, (_, te) in enumerate(folds):
        d = pd.DataFrame({"row_id": ids_tr[te], "evaluacion": "validacion_r0", "pliegue": f, "y": y[ids_tr[te]]})
        com = pickle.load(open(crudos / "telco" / f"r0_f{f}" / "_combinadores.pkl", "rb"))
        for m in MODELOS:
            e = com[m] if m == ORIGINAL else pickle.load(open(crudos / "telco" / f"r0_f{f}" / f"{m}.pkl", "rb"))
            d[m], d[f"u_{m}"] = e["pte"], e["umbral"]
        filas.append(d)
    d = pd.DataFrame({"row_id": ids_te, "evaluacion": "holdout", "pliegue": -1, "y": y[ids_te]})
    com = pickle.load(open(crudos / "telco_holdout" / "r9_f0" / "_combinadores.pkl", "rb"))
    historia = {}
    for m in MODELOS:
        e = com[m] if m == ORIGINAL else pickle.load(open(crudos / "telco_holdout" / "r9_f0" / f"{m}.pkl", "rb"))
        d[m], d[f"u_{m}"] = e["pte"], e["umbral"]
        historia[m] = [float(t["valor"]) for t in e["registro"]]
    filas.append(d)
    P = pd.concat(filas, ignore_index=True)
    umbrales = P.filter(like="u_").groupby(P.evaluacion.astype(str) + "_" + P.pliegue.astype(str)).first()
    P.drop(columns=[c for c in P if c.startswith("u_")]).to_csv(SAL / "predicciones.csv", index=False, float_format="%.6f")
    umbrales.to_csv(SAL / "umbrales.csv")
    r = com[ORIGINAL]
    conv = {"log_posterior": [float(v) for v in r["historial"]], "n_iter": int(r["n_iter"]), "convergio": bool(r["convergio"]),
            "params": r["params"], "busqueda_log_loss_interno": historia}
    (SAL / "convergencia.json").write_text(json.dumps(conv, ensure_ascii=False), encoding="utf-8")


def _shap_trozo(modelo, filas, semilla):
    from src.modelos import shap_local
    from src.pipeline_telco import ENTRADA as E
    fondo = fondo_global()
    out = []
    for _, fila in filas.iterrows():
        v, base = shap_local(modelo, fila[E].to_frame().T.astype(fondo.dtypes.to_dict()), fondo, 200, semilla + int(fila.row_id))
        out.append({"modelo": modelo, "row_id": int(fila.row_id), "base": base, **v.to_dict()})
    return out


def fondo_global():
    raw = pd.read_csv(RAIZ / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    ids = pd.read_csv(RAIZ / "data" / "train_index.csv")["train_index"].to_numpy()
    X = raw.loc[ids, ENTRADA].reset_index(drop=True)
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
    return X.iloc[np.sort(np.random.default_rng(seed_for("test", 910)).choice(len(X), 50, replace=False))].reset_index(drop=True)


def shap_global(n):
    """SHAP de 150 clientes fijos del hold-out para cada modelo (50 de referencia, 200 coaliciones)."""
    raw = pd.read_csv(RAIZ / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    ids = pd.read_csv(RAIZ / "data" / "test_index.csv")["test_index"].to_numpy()
    sel = np.sort(np.random.default_rng(seed_for("test", 911)).choice(ids, 150, replace=False))
    X = raw.loc[sel, ENTRADA].copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
    X.insert(0, "row_id", sel)
    trozos = [(m, t) for m in MODELOS for t in np.array_split(X, 10)]
    partes = Parallel(n_jobs=n, batch_size=1)(delayed(_shap_trozo)(m, t, seed_for("test", 912) % 2**31) for m, t in trozos)
    pd.DataFrame([r for p in partes for r in p]).to_csv(SAL / "shap_global.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    proyecto = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    procedencia = copiar(proyecto)
    predicciones(proyecto)
    shap_global(n)
    (SAL / "procedencia.json").write_text(json.dumps(procedencia, ensure_ascii=False, indent=1), encoding="utf-8")
    print("listo")
