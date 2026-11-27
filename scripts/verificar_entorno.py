"""Comprueba versiones y que los 6 modelos carguen y predigan. Uso: python scripts/verificar_entorno.py"""
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
esperadas = {"xgboost": "3.4.1", "scikit-learn": "1.9.1", "pandas": "3.0.5", "numpy": "2.5.3", "streamlit": "1.67.0"}
for p, v in esperadas.items():
    try:
        actual = version(p)
    except Exception:
        actual = "NO INSTALADO"
    print(f"{p:14s} instalada {actual:14s} esperada {v} {'OK' if actual == v else '<-- DIFERENTE: pip install -r requirements.txt'}")
from rendimiento import core

for t in ("reg", "clf"):
    for e in core.ESCENARIOS:
        core.modelo(t, e)
r = core.predecir("Con G1 y G2", {"G1": 10, "G2": 11})
print("Modelos cargados y predicción de prueba:", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
