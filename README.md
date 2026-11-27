# Rendimiento académico en lengua portuguesa: dashboard y proyecto integrador

Proyecto Integrador de Ciencia de Datos (Maestría en Inteligencia Artificial, Universidad de La Salle). Autor: Andrés Felipe Ricaurte Cortés.

El repositorio contiene un dashboard interactivo (Streamlit y Plotly) que muestra el panorama de 649 estudiantes de secundaria, predice la nota final y el riesgo de reprobar en tres escenarios (sin notas, con G1, con G1 y G2) y segmenta a los estudiantes en cuatro perfiles.

## Ejecución rápida

Requiere Python 3.12 o superior.

    ./run.sh            # Linux y macOS
    run.bat             # Windows

La primera vez crea el entorno virtual `.venv` e instala `requirements.txt` (con uv si está instalado, que es mucho más rápido). Después abre http://localhost:8501.

Instalación manual:

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run dashboard/app.py

## Estructura

    data/                  datos originales (student-por.csv) y preparados (data/processed)
    modelos/               modelos finales y segmentación de la Fase 3 (joblib)
    src/rendimiento/       lógica compartida: datos, predicción y segmentación
    dashboard/app.py       aplicación de Streamlit
    notebooks/             cuadernos de las cuatro fases (el de la Fase 4 es el informe técnico)
    tests/                 pruebas automáticas (pytest)
    scripts/               exportar_xgboost.py y verificar_entorno.py
    requirements.txt       dependencias del dashboard con versiones exactas
    requirements-notebooks.txt   dependencias para reproducir los cuadernos (PyTorch, SHAP, etc.)

## Pruebas

    pip install pytest
    pytest -q tests

Las pruebas comprueban que los modelos reproducen las métricas de prueba de la Fase 3 y que el dashboard arranca y responde a los controles.

## Despliegue en vivo (Streamlit Community Cloud)

1. Sube esta carpeta a un repositorio público de GitHub.
2. En https://share.streamlit.io elige el repositorio, la rama y el archivo principal `dashboard/app.py`.
3. La plataforma instala `requirements.txt` y publica una URL pública.

## Notas

Los modelos se guardaron con scikit-learn 1.9.1 y XGBoost 3.4.1; usa esas versiones (están fijadas en requirements.txt) porque los archivos joblib no son portables entre versiones distintas. El conjunto de datos no tiene variable de fecha, por lo que los filtros del dashboard son por rangos numéricos y categorías.

Datos: Cortez, P. y Silva, A. (2008), Student Performance, UCI Machine Learning Repository, CC BY 4.0, https://doi.org/10.24432/C5TG7T

## Si aparece un error al cargar los modelos

Un error como `XGBoostError: input stream corrupted` indica casi siempre una versión distinta de xgboost o un archivo alterado al copiarlo. Ejecuta `python scripts/verificar_entorno.py`: muestra qué versiones difieren y si los modelos cargan. Los modelos XGBoost se guardan además en JSON (modelos/*.json), que es portable entre versiones; si clonas con git en Windows, el archivo `.gitattributes` evita que se alteren los binarios.
