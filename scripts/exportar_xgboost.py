"""Guarda los modelos XGBoost en formato JSON (portable entre versiones) y deja el pipeline sin el booster.

Se ejecuta una sola vez con las versiones con las que se entrenó (xgboost 3.4.1). El dashboard carga después
el JSON, por lo que ya no depende de que el pickle de XGBoost sea idéntico en cada equipo.
"""
import sys
from pathlib import Path

import joblib

M = Path(__file__).resolve().parents[1] / "modelos"
for f in sorted(M.glob("*_*.joblib")):
    if f.name.startswith("segmentacion"):
        continue
    pipe = joblib.load(f)
    est = pipe.named_steps["m"]
    if type(est).__module__.split(".")[0] != "xgboost" or not hasattr(est, "get_booster"):
        continue
    est.save_model(M / (f.stem + ".json"))
    pipe.steps[-1] = ("m", type(est)(**est.get_params()))  # estimador sin entrenar: se rellena desde el JSON
    joblib.dump(pipe, f)
    print("exportado", f.stem)
