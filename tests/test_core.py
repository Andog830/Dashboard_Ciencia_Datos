import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rendimiento import core

COLS = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup",
        "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]


def test_datos_concuerdan_con_fases_anteriores():
    df = core.datos()
    assert df.shape[0] == 649 and df.aprobado.sum() == 549 and (df.G3 == 0).sum() == 15
    assert df.particion.value_counts().to_dict() == {"entrenamiento": 454, "validación": 97, "prueba": 98}
    tasa = (1 - df.groupby("grupo").aprobado.mean()).values
    assert list(np.argsort(tasa)) == [0, 1, 2, 3]  # grupos ordenados por tasa de reprobación


@pytest.mark.parametrize("esc", core.ESCENARIOS)
def test_modelos_reproducen_resultados_de_prueba(esc):
    te = core.datos().query("particion == 'prueba'")
    import pandas as pd
    X = pd.concat([core.derivar({k: (core.valores_tipicos()["G1"] if k == "G1" and np.isnan(r[k]) else r[k]) for k in COLS}) for _, r in te.iterrows()], ignore_index=True)
    ok = (te.G3 > 0).values
    rmse = mean_squared_error(te.G3[ok], core.modelo("reg", esc).predict(X[ok])) ** 0.5
    auc = roc_auc_score(1 - te.aprobado, core.modelo("clf", esc).predict_proba(X)[:, 1])
    assert rmse == pytest.approx(core.resultados()["test_reg"][esc]["RMSE"], abs=1e-6)
    assert auc == pytest.approx(core.resultados()["test_clf"][esc]["AUC"], abs=1e-6)


def test_prediccion_y_grupo_validos():
    r = core.predecir("Con G1 y G2", {"G1": 6, "G2": 5, "failures": 2})
    assert 0 <= r["nota"] <= 20 and 0 <= r["p_reprobar"] <= 1 and r["riesgo"] in {"Bajo", "Medio", "Alto"}
    assert core.predecir("Con G1 y G2", {"G1": 16, "G2": 17})["p_reprobar"] < r["p_reprobar"]
    assert core.asignar_grupo({}) in {1, 2, 3, 4}
    assert core.nivel_riesgo(0.9, 0.5) == "Alto" and core.nivel_riesgo(0.3, 0.5) == "Medio" and core.nivel_riesgo(0.1, 0.5) == "Bajo"
