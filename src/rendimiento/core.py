"""Lógica compartida por el dashboard y el cuaderno de la Fase 4.

Carga los datos preparados en la Fase 2, los modelos finales y la segmentación de la Fase 3,
y ofrece funciones para predecir y asignar grupos a un estudiante nuevo.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data" / "processed"
MODELOS = RAIZ / "modelos"

ESCENARIOS = ["Sin notas", "Con G1", "Con G1 y G2"]
ORDINALES = ["Medu", "Fedu", "traveltime", "studytime", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]

# Resultados de los modelos base de la Fase 3 en el conjunto de prueba (tablas 12 y 13 del cuaderno de la Fase 3)
BASE_PRUEBA = {
    "Sin notas": {"RMSE": 2.452, "R2": 0.122, "F1": 0.400, "AUC": 0.680},
    "Con G1": {"RMSE": 1.305, "R2": 0.751, "F1": 0.524, "AUC": 0.900},
    "Con G1 y G2": {"RMSE": 0.833, "R2": 0.899, "F1": 0.714, "AUC": 0.976},
}

# Nombres de los grupos de K-Means, ordenados de menor a mayor tasa de reprobación (Fase 3, sección 9.3)
GRUPOS = {
    1: ("Perfil académico sólido", "Hogares con mayor nivel educativo, consumo de alcohol bajo, más tiempo de estudio y casi todos quieren estudios superiores."),
    2: ("Hábitos moderados", "Hábitos parecidos a los del grupo 1 pero con hogares de menor nivel educativo y mayor presencia de la escuela MS."),
    3: ("Vida social intensa", "Muchas salidas con amigos, el mayor consumo de alcohol (sobre todo en fin de semana), menos estudio y más ausencias."),
    4: ("Rezago y riesgo alto", "Estudiantes mayores, con reprobaciones previas, menos interés en estudios superiores y la mayor tasa de reprobación."),
}


@lru_cache(maxsize=1)
def config() -> dict:
    return json.loads((DATOS / "config_fase2.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def resultados() -> dict:
    return json.loads((DATOS / "resultados_fase3.json").read_text(encoding="utf-8"))


def columnas(escenario: str) -> list[str]:
    base = config()["cols_sin_notas"]
    return {"Sin notas": base, "Con G1": base + ["G1"], "Con G1 y G2": config()["cols_completo"]}[escenario]


@lru_cache(maxsize=1)
def datos() -> pd.DataFrame:
    """Datos de la Fase 2 (649 estudiantes) con el grupo de K-Means de la Fase 3."""
    df = pd.read_csv(DATOS / "student_por_fase2.csv", index_col="fila")
    df = df.join(pd.read_csv(DATOS / "grupos_fase3.csv", index_col="fila"))
    df["estado"] = np.where(df["aprobado"] == 1, "Aprobado", "Reprobado")
    return df


@lru_cache(maxsize=1)
def valores_tipicos() -> dict:
    """Mediana de las variables numéricas y moda de las categóricas, usadas como valores iniciales."""
    df = datos()
    out = {}
    for c in config()["nominales"]:
        out[c] = df[c].mode().iloc[0]
    for c in ["age", "failures", "absences", "G1", "G2"] + ORDINALES:
        out[c] = int(round(df[c].median()))
    return out


def derivar(entrada: dict) -> pd.DataFrame:
    """Convierte las respuestas de un estudiante en una fila con las variables que esperan los modelos."""
    d = {**valores_tipicos(), **entrada}
    d["alcohol_total"] = d["Dalc"] + d["Walc"]
    d["educ_padres"] = (d["Medu"] + d["Fedu"]) / 2
    d["tiene_reprob"] = int(d["failures"] > 0)
    d["log_ausencias"] = float(np.log1p(d["absences"]))
    d["tendencia_notas"] = d["G2"] - d["G1"]
    d["apoyo_total"] = sum(int(d[k] == "yes") for k in ("schoolsup", "famsup", "paid"))
    return pd.DataFrame([d])


@lru_cache(maxsize=None)
def modelo(tarea: str, escenario: str):
    """Carga el pipeline; si el modelo es XGBoost, su booster se lee del JSON portable que lo acompaña."""
    base = MODELOS / f"{tarea}_{escenario.replace(' ', '_')}"
    pipe = joblib.load(base.with_suffix(".joblib"))
    if base.with_suffix(".json").exists():
        pipe.named_steps["m"].load_model(base.with_suffix(".json"))
    return pipe


def nivel_riesgo(p: float, umbral: float) -> str:
    """Bajo por debajo de la mitad del umbral, medio hasta el umbral y alto desde el umbral."""
    return "Alto" if p >= umbral else ("Medio" if p >= umbral / 2 else "Bajo")


def predecir(escenario: str, entrada: dict) -> dict:
    X = derivar(entrada)
    nota = float(np.clip(modelo("reg", escenario).predict(X)[0], 0, 20))
    p = float(modelo("clf", escenario).predict_proba(X)[0, 1])
    umbral = float(resultados()["umbrales"][escenario])
    rmse = float(resultados()["test_reg"][escenario]["RMSE"])
    return {"nota": nota, "rmse": rmse, "p_reprobar": p, "umbral": umbral, "riesgo": nivel_riesgo(p, umbral)}


@lru_cache(maxsize=1)
def segmentacion() -> dict:
    return joblib.load(MODELOS / "segmentacion.joblib")


def _escalar(df: pd.DataFrame) -> np.ndarray:
    s = segmentacion()
    return s["escalador"].transform(df[s["variables"]])


def asignar_grupo(entrada: dict) -> int:
    s = segmentacion()
    etiqueta = int(s["kmeans"].predict(_escalar(derivar(entrada)))[0])
    return int(s["mapa_grupos"][etiqueta])


@lru_cache(maxsize=1)
def _pca():
    from sklearn.decomposition import PCA
    df = datos()
    pca = PCA(n_components=3, random_state=config()["semilla"]).fit(_escalar(df[df["particion"] == "entrenamiento"]))
    return pca


@lru_cache(maxsize=1)
def proyeccion() -> pd.DataFrame:
    """Coordenadas de los 649 estudiantes en las tres primeras componentes principales."""
    df = datos()
    return pd.DataFrame(_pca().transform(_escalar(df)), index=df.index, columns=["PC1", "PC2", "PC3"])


def proyectar(entrada: dict) -> np.ndarray:
    return _pca().transform(_escalar(derivar(entrada)))[0]


def varianza_explicada() -> np.ndarray:
    return _pca().explained_variance_ratio_


def perfil_estandarizado(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Promedio estandarizado de las variables de segmentación en cada grupo."""
    df = datos() if df is None else df
    s = segmentacion()
    z = pd.DataFrame(_escalar(df), index=df.index, columns=s["variables"])
    return z.groupby(df["grupo"]).mean()
