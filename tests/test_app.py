from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_app_arranca_y_responde_a_controles():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5 and len(at.metric) >= 9
    for esc in ["Con G1", "Con G1 y G2", "Sin notas"]:
        at.sidebar.radio[0].set_value(esc).run()
        assert not at.exception, (esc, at.exception)
    at.sidebar.radio[0].set_value("Con G1 y G2").run()
    at.slider(key="p_g1").set_value(6).run()
    at.slider(key="p_g2").set_value(4).run()
    assert not at.exception and any(m.label == "Nivel de riesgo" and m.value == "Alto" for m in at.metric)
    at.slider(key="p_g1").set_value(17).run()
    at.slider(key="p_g2").set_value(18).run()
    assert any(m.label == "Nivel de riesgo" and m.value == "Bajo" for m in at.metric)
    at.sidebar.multiselect[1].set_value(["MS"]).run()  # filtro de escuela
    assert not at.exception


def test_filtros_sin_resultados_avisan():
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.slider[2].set_value((3, 3)).run()   # reprobaciones previas: solo 3
    at.sidebar.slider[3].set_value((15, 20)).run()  # y nota final alta: no hay estudiantes así
    assert not at.exception and len(at.warning) == 1
