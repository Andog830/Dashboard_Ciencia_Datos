@echo off
REM Crea un entorno virtual la primera vez, instala dependencias y lanza el dashboard.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\python -m pip install --quiet -r requirements.txt
)
.venv\Scripts\python -m streamlit run dashboard\app.py %*
