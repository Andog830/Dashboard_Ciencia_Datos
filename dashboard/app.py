"""Dashboard interactivo del Proyecto Integrador (Fase 4): rendimiento en lengua portuguesa."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rendimiento import core

st.set_page_config(page_title="Rendimiento académico", layout="wide", initial_sidebar_state="expanded")

AZUL, NARANJA, VERDE, AMARILLO = "#0072B2", "#D55E00", "#009E73", "#E69F00"
COLOR_GRUPO = {"Grupo 1": AZUL, "Grupo 2": VERDE, "Grupo 3": AMARILLO, "Grupo 4": NARANJA}
COLOR_ESTADO = {"Aprobado": AZUL, "Reprobado": NARANJA}
SI_NO = {"yes": "Sí", "no": "No"}
ETIQ = {
    "school": "Escuela", "sex": "Sexo", "address": "Zona de residencia", "higher": "Quiere educación superior", "famsize": "Tamaño de la familia",
    "Pstatus": "Convivencia de los padres", "Mjob": "Ocupación de la madre", "Fjob": "Ocupación del padre", "reason": "Motivo de elección de la escuela",
    "guardian": "Tutor legal", "schoolsup": "Apoyo de la escuela", "famsup": "Apoyo de la familia", "paid": "Clases pagas", "activities": "Actividades extracurriculares",
    "nursery": "Asistió a guardería", "internet": "Internet en casa", "romantic": "Relación sentimental", "failures": "Reprobaciones previas",
    "studytime": "Tiempo de estudio", "Medu": "Educación de la madre", "Fedu": "Educación del padre", "traveltime": "Tiempo de viaje", "famrel": "Relaciones familiares",
    "freetime": "Tiempo libre", "goout": "Salidas con amigos", "Dalc": "Alcohol entre semana", "Walc": "Alcohol fin de semana", "health": "Salud", "age": "Edad",
    "absences": "Ausencias", "G1": "Nota 1.er periodo", "G2": "Nota 2.º periodo", "G3": "Nota final", "alcohol_total": "Alcohol total", "educ_padres": "Educación de los padres",
}
df = core.datos()
tip = core.valores_tipicos()
res = core.resultados()


def graficar(fig, **kw):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13), legend_title_text="")
    st.plotly_chart(fig, width="stretch", **kw)


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("Rendimiento académico")
    st.caption("Estudiantes de secundaria en Portugal, lengua portuguesa (649 registros).")
    esc = st.radio("Escenario del modelo", core.ESCENARIOS, index=0,
                   help="Sin notas: solo contexto y hábitos (disponible al inicio del año). Con G1: después del primer periodo. Con G1 y G2: después del segundo.")
    st.divider()
    st.subheader("Filtros")
    st.caption("El conjunto no tiene variable de fecha, por eso se filtra por rangos numéricos y categorías.")
    particion = st.multiselect("Partición de los datos", ["entrenamiento", "validación", "prueba"], default=["entrenamiento", "validación", "prueba"])
    escuela = st.multiselect("Escuela", ["GP", "MS"], default=["GP", "MS"])
    sexo = st.multiselect("Sexo", ["F", "M"], default=["F", "M"])
    zona = st.multiselect("Zona", ["U", "R"], default=["U", "R"], format_func=lambda x: {"U": "Urbana", "R": "Rural"}[x])
    sup = st.multiselect("Quiere educación superior", ["yes", "no"], default=["yes", "no"], format_func=SI_NO.get)
    edad = st.slider("Edad", int(df.age.min()), int(df.age.max()), (int(df.age.min()), int(df.age.max())))
    aus = st.slider("Ausencias", 0, int(df.absences.max()), (0, int(df.absences.max())))
    rep = st.slider("Reprobaciones previas", 0, int(df.failures.max()), (0, int(df.failures.max())))
    g3 = st.slider("Nota final (G3)", 0, 20, (0, 20))
    grupos = st.multiselect("Grupo de segmentación", [1, 2, 3, 4], default=[1, 2, 3, 4], format_func=lambda g: f"{g}. {core.GRUPOS[g][0]}")

mask = (df.particion.isin(particion) & df.school.isin(escuela) & df.sex.isin(sexo) & df.address.isin(zona) & df.higher.isin(sup)
        & df.age.between(*edad) & df.absences.between(*aus) & df.failures.between(*rep) & df.G3.between(*g3) & df.grupo.isin(grupos))
dff = df[mask]

st.title("Panel de rendimiento y riesgo de reprobación")
if dff.empty:
    st.warning("Ningún estudiante cumple los filtros seleccionados. Amplía los rangos en la barra lateral.")
    st.stop()

tab_pan, tab_pred, tab_seg, tab_mod, tab_dat = st.tabs(["Panorama", "Predicción en vivo", "Segmentación", "Modelos", "Datos y acerca de"])

# ------------------------------------------------------------------ panorama
with tab_pan:
    reales = dff[dff.G3 > 0]
    k = st.columns(5)
    k[0].metric("Estudiantes", f"{len(dff):,}", f"{len(dff) / len(df) * 100:.0f} % del total", delta_color="off", border=True)
    k[1].metric("Tasa de aprobación", f"{dff.aprobado.mean() * 100:.1f} %", help="Aprobado: nota final de 10 o más", border=True)
    k[2].metric("Reprobados", f"{int((dff.aprobado == 0).sum())}", border=True)
    k[3].metric("Nota final media", f"{reales.G3.mean():.2f}" if len(reales) else "-", help="Sin contar los estudiantes con nota final igual a cero", border=True)
    k[4].metric("Ausencias medias", f"{dff.absences.mean():.1f}", border=True)

    rt, rc = res["test_reg"][esc], res["test_clf"][esc]
    st.caption(f"Desempeño del modelo del escenario «{esc}» en el conjunto de prueba (98 estudiantes)")
    m = st.columns(4)
    m[0].metric("R² (nota final)", f"{rt['R2']:.2f}", border=True)
    m[1].metric("Error típico (RMSE)", f"{rt['RMSE']:.2f} puntos", border=True)
    m[2].metric("F1 de reprobado", f"{rc['F1']:.2f}", help="Con 15 reprobados en la prueba, esta cifra es poco estable", border=True)
    m[3].metric("AUC", f"{rc['AUC']:.2f}", border=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x="G3", color="estado", nbins=20, color_discrete_map=COLOR_ESTADO, barmode="overlay", opacity=0.8,
                           labels={"G3": "Nota final", "count": "Estudiantes"}, title="Distribución de la nota final")
        fig.add_vline(x=10, line_dash="dash", line_color=NARANJA, annotation_text="Mínimo para aprobar")
        graficar(fig)
    with c2:
        opciones = [v for v in ETIQ if v in dff.columns and v not in ("G1", "G2", "G3", "absences", "age", "alcohol_total", "educ_padres")]
        var = st.selectbox("Comparar la reprobación según", opciones, index=opciones.index("higher"), format_func=ETIQ.get)
        t = dff.groupby(var).agg(n=("aprobado", "size"), reprobados=("aprobado", lambda s: (1 - s).mean() * 100)).reset_index()
        t[var] = t[var].astype(str).map(lambda x: SI_NO.get(x, x))
        fig = px.bar(t, x=var, y="reprobados", text=t.apply(lambda r: f"{r.reprobados:.0f} % (n={int(r.n)})", axis=1), color_discrete_sequence=[AZUL],
                     labels={var: ETIQ[var], "reprobados": "Reprobados (%)"}, title=f"Reprobación según: {ETIQ[var].lower()}")
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, max(10, t.reprobados.max() * 1.25)])
        graficar(fig)

    c3, c4 = st.columns(2)
    with c3:
        xv = st.selectbox("Relación de la nota final con", ["G2", "G1", "absences", "age", "alcohol_total", "educ_padres", "studytime", "failures"], format_func=ETIQ.get)
        rng = np.random.default_rng(42)
        d = dff.assign(_x=dff[xv] + rng.uniform(-0.15, 0.15, len(dff)), _y=dff.G3 + rng.uniform(-0.15, 0.15, len(dff)))
        fig = px.scatter(d, x="_x", y="_y", color="estado", color_discrete_map=COLOR_ESTADO, opacity=0.6, hover_data={"_x": False, "_y": False, xv: True, "G3": True},
                         labels={"_x": ETIQ[xv], "_y": "Nota final"}, title=f"Nota final y {ETIQ[xv].lower()}")
        graficar(fig)
    with c4:
        num = ["age", "failures", "absences", "studytime", "Medu", "Fedu", "Dalc", "Walc", "goout", "G1", "G2", "G3"]
        cor = reales[num].corr(method="spearman")
        cor.index = cor.columns = [ETIQ[c] for c in num]
        fig = px.imshow(cor, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto", title="Correlación de Spearman (sin ceros en la nota final)")
        graficar(fig)

# ------------------------------------------------------------------ predicción en vivo
with tab_pred:
    st.subheader("Simulador de predicción")
    st.caption("Ajusta los datos de un estudiante y el modelo del escenario elegido en la barra lateral calcula al instante la nota final esperada y el riesgo de reprobar.")
    ci, cr = st.columns([2, 3])
    with ci:
        e = {}
        a, b = st.columns(2)
        e["school"] = a.selectbox("Escuela", ["GP", "MS"], index=["GP", "MS"].index(tip["school"]), key="p_school")
        e["sex"] = b.selectbox("Sexo", ["F", "M"], index=["F", "M"].index(tip["sex"]), key="p_sex")
        e["age"] = a.slider("Edad", 15, 22, tip["age"], key="p_age")
        e["address"] = b.selectbox("Zona", ["U", "R"], index=["U", "R"].index(tip["address"]), format_func=lambda x: {"U": "Urbana", "R": "Rural"}[x], key="p_addr")
        e["higher"] = a.selectbox("Quiere educación superior", ["yes", "no"], index=["yes", "no"].index(tip["higher"]), format_func=SI_NO.get, key="p_high")
        e["failures"] = b.slider("Reprobaciones previas", 0, 3, tip["failures"], key="p_fail")
        e["absences"] = a.slider("Ausencias", 0, 32, tip["absences"], key="p_abs")
        e["studytime"] = b.select_slider("Tiempo de estudio semanal", [1, 2, 3, 4], value=tip["studytime"], format_func=lambda x: {1: "< 2 h", 2: "2 a 5 h", 3: "5 a 10 h", 4: "> 10 h"}[x], key="p_study")
        if esc in ("Con G1", "Con G1 y G2"):
            e["G1"] = a.slider("Nota del primer periodo (G1)", 0, 20, tip["G1"], key="p_g1")
        if esc == "Con G1 y G2":
            e["G2"] = b.slider("Nota del segundo periodo (G2)", 0, 20, tip["G2"], key="p_g2")
        with st.expander("Más datos del estudiante (opcional)"):
            a, b = st.columns(2)
            for i, v in enumerate(["Medu", "Fedu"]):
                e[v] = (a, b)[i].slider(ETIQ[v], 0, 4, tip[v], key=f"p_{v}")
            e["traveltime"] = a.slider(ETIQ["traveltime"], 1, 4, tip["traveltime"], key="p_tt")
            for i, v in enumerate(["famrel", "freetime", "goout", "Dalc", "Walc", "health"]):
                e[v] = (b, a)[i % 2].slider(ETIQ[v], 1, 5, tip[v], key=f"p_{v}")
            for i, v in enumerate(["Mjob", "Fjob", "reason", "guardian"]):
                op = sorted(df[v].unique())
                e[v] = (a, b)[i % 2].selectbox(ETIQ[v], op, index=op.index(tip[v]), key=f"p_{v}")
            for i, v in enumerate(["famsize", "Pstatus"]):
                op = sorted(df[v].unique())
                e[v] = (a, b)[i % 2].selectbox(ETIQ[v], op, index=op.index(tip[v]), key=f"p_{v}")
            for i, v in enumerate(["schoolsup", "famsup", "paid", "activities", "nursery", "internet", "romantic"]):
                e[v] = (a, b)[i % 2].selectbox(ETIQ[v], ["yes", "no"], index=["yes", "no"].index(tip[v]), format_func=SI_NO.get, key=f"p_{v}")
    r = core.predecir(esc, e)
    grupo = core.asignar_grupo(e)
    with cr:
        a, b, c = st.columns(3)
        a.metric("Nota final esperada", f"{r['nota']:.1f}", f"± {r['rmse']:.1f} (error típico)", delta_color="off", border=True)
        b.metric("Probabilidad de reprobar", f"{r['p_reprobar'] * 100:.0f} %", border=True)
        c.metric("Nivel de riesgo", r["riesgo"], help=f"Alto desde {r['umbral'] * 100:.0f} %, medio desde {r['umbral'] * 50:.0f} %", border=True)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=r["p_reprobar"] * 100, number={"suffix": " %"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333333"},
                                            "steps": [{"range": [0, r["umbral"] * 50], "color": "#b9e3d4"}, {"range": [r["umbral"] * 50, r["umbral"] * 100], "color": "#f6d99a"},
                                                      {"range": [r["umbral"] * 100, 100], "color": "#f0b999"}]},
                                     title={"text": "Riesgo de reprobar"}))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        nombre, desc = core.GRUPOS[grupo]
        st.info(f"Por sus hábitos y contexto, este estudiante se parece al grupo {grupo}: {nombre}. {desc}")
        if esc == "Sin notas" or r["riesgo"] != "Bajo":
            st.caption("Lectura prudente: con estos datos el modelo ordena el riesgo de forma modesta y sirve para priorizar acompañamiento, no para decidir sobre un estudiante concreto.")
        st.caption("La nota esperada se calculó con estudiantes que sí tuvieron nota final; no anticipa abandonos (nota final igual a cero).")

# ------------------------------------------------------------------ segmentación
with tab_seg:
    st.subheader("Segmentación por hábitos y contexto (K-Means, 4 grupos)")
    st.caption("Los grupos se formaron sin usar las notas. Se muestran los estudiantes que cumplen los filtros.")
    modo = st.radio("Vista", ["2D", "3D"], horizontal=True)
    P = core.proyeccion().loc[dff.index]
    ve = core.varianza_explicada()
    pl = P.assign(Grupo=dff.grupo.map(lambda g: f"Grupo {g}"), Nota=dff.G3, Estado=dff.estado).sort_values("Grupo")
    kw = dict(color="Grupo", color_discrete_map=COLOR_GRUPO, hover_data={"Nota": True, "Estado": True, "PC1": ":.2f", "PC2": ":.2f"}, category_orders={"Grupo": list(COLOR_GRUPO)})
    pc = core.proyectar(e)
    if modo == "2D":
        fig = px.scatter(pl, x="PC1", y="PC2", opacity=0.65, labels={"PC1": f"Componente 1 ({ve[0] * 100:.0f} %)", "PC2": f"Componente 2 ({ve[1] * 100:.0f} %)"}, **kw)
        fig.add_trace(go.Scatter(x=[pc[0]], y=[pc[1]], mode="markers", name="Estudiante simulado", marker=dict(symbol="star", size=18, color="black", line=dict(color="white", width=1))))
    else:
        kw["hover_data"] = {"Nota": True, "Estado": True}
        fig = px.scatter_3d(pl, x="PC1", y="PC2", z="PC3", opacity=0.7, **kw)
        fig.update_traces(marker_size=3)
        fig.add_trace(go.Scatter3d(x=[pc[0]], y=[pc[1]], z=[pc[2]], mode="markers", name="Estudiante simulado", marker=dict(symbol="diamond", size=7, color="black")))
        fig.update_layout(height=560)
    graficar(fig)

    z = core.perfil_estandarizado(dff)
    z.index = [f"Grupo {g}" for g in z.index]
    z.columns = [ETIQ.get(c, "Ausencias (log)" if c == "log_ausencias" else c) for c in z.columns]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.imshow(z.T, zmin=-1.2, zmax=1.2, color_continuous_scale="RdBu_r", text_auto=".1f", aspect="auto", title="Perfil de cada grupo (desviaciones estándar)")
        graficar(fig)
    with c2:
        tp = dff.groupby("grupo").agg(Estudiantes=("G3", "size"), Reprobados=("aprobado", lambda s: round((1 - s).mean() * 100, 1)),
                                      Nota_media=("G3", lambda s: round(s[s > 0].mean(), 2) if (s > 0).any() else np.nan), Edad=("age", "mean"), Ausencias=("absences", "mean")).round(1)
        tp.index = [f"{g}. {core.GRUPOS[g][0]}" for g in tp.index]
        tp.columns = ["Estudiantes", "Reprobados (%)", "Nota media", "Edad media", "Ausencias medias"]
        st.markdown("Resumen por grupo")
        st.dataframe(tp, width="stretch")
        st.caption("Los grupos están numerados de menor a mayor tasa de reprobación. La silueta de K-Means fue baja (0,10): los estudiantes forman un continuo y los grupos son una partición útil, no categorías nítidas.")
    with st.expander("Descripción de los grupos"):
        for g, (n, d_) in core.GRUPOS.items():
            st.markdown(f"Grupo {g}, {n}: {d_}")

# ------------------------------------------------------------------ modelos
with tab_mod:
    st.subheader("Modelos finales y resultados en el conjunto de prueba")
    filas = []
    for s in core.ESCENARIOS:
        rt, rc, bb = res["test_reg"][s], res["test_clf"][s], core.BASE_PRUEBA[s]
        filas.append({"Escenario": s, "Regresión": res["elegidos"][f"reg|{s}"], "RMSE": rt["RMSE"], "RMSE base": bb["RMSE"], "R²": rt["R2"], "R² base": bb["R2"],
                      "Clasificación": res["elegidos"][f"clf|{s}"], "Umbral": res["umbrales"][s], "F1": rc["F1"], "F1 base": bb["F1"], "AUC": rc["AUC"], "AUC base": bb["AUC"]})
    tm = pd.DataFrame(filas).set_index("Escenario")
    st.dataframe(tm.style.format({c: "{:.2f}" for c in tm.select_dtypes("number").columns}), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        d = pd.DataFrame({"Escenario": core.ESCENARIOS * 2, "Modelo": ["Final"] * 3 + ["Base"] * 3,
                          "RMSE": [res["test_reg"][s]["RMSE"] for s in core.ESCENARIOS] + [core.BASE_PRUEBA[s]["RMSE"] for s in core.ESCENARIOS]})
        fig = px.bar(d, x="Escenario", y="RMSE", color="Modelo", barmode="group", text_auto=".2f", color_discrete_map={"Final": AZUL, "Base": "#B0B0B0"}, title="Regresión: error en prueba (menor es mejor)")
        graficar(fig)
    with c2:
        d = pd.DataFrame({"Escenario": core.ESCENARIOS * 2, "Modelo": ["Final"] * 3 + ["Base"] * 3,
                          "F1": [res["test_clf"][s]["F1"] for s in core.ESCENARIOS] + [core.BASE_PRUEBA[s]["F1"] for s in core.ESCENARIOS]})
        fig = px.bar(d, x="Escenario", y="F1", color="Modelo", barmode="group", text_auto=".2f", color_discrete_map={"Final": AZUL, "Base": "#B0B0B0"}, title="Clasificación: F1 de reprobado en prueba (mayor es mejor)")
        graficar(fig)
    st.warning("La regresión mejora con claridad al modelo base solo sin notas (error 11 % menor). En clasificación los modelos finales no superaron de forma consistente a los base; "
               "la prueba tiene solo 15 reprobados y los intervalos de confianza son muy amplios. Se recomienda leer la probabilidad como nivel de riesgo y no como una etiqueta definitiva.")

# ------------------------------------------------------------------ datos y acerca de
with tab_dat:
    st.subheader("Datos filtrados")
    cols = ["school", "sex", "age", "address", "higher", "failures", "absences", "studytime", "G1", "G2", "G3", "estado", "grupo", "particion"]
    st.dataframe(dff[cols], width="stretch", height=320)
    st.download_button("Descargar los datos filtrados (CSV)", dff.to_csv().encode("utf-8"), "estudiantes_filtrados.csv", "text/csv")
    st.subheader("Acerca del proyecto")
    st.markdown(
        "Proyecto Integrador de Ciencia de Datos, Maestría en Inteligencia Artificial, Universidad de La Salle (Bogotá). Autor: Andrés Felipe Ricaurte Cortés.\n\n"
        "Datos: Student Performance, archivo de lengua portuguesa, UCI Machine Learning Repository (Cortez y Silva, 2008), licencia CC BY 4.0. "
        "Las fases anteriores (exploración, preparación y modelado) están en la carpeta notebooks del repositorio.\n\n"
        "Es una herramienta académica: las predicciones describen asociaciones en dos escuelas de Portugal y no son causas ni deben usarse para decisiones sobre estudiantes concretos.")
