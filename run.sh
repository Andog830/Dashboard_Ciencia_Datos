#!/usr/bin/env bash
# Crea un entorno virtual la primera vez, instala dependencias (con uv si está disponible) y lanza el dashboard.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  if command -v uv >/dev/null 2>&1; then uv pip install --python .venv/bin/python -r requirements.txt
  else .venv/bin/python -m pip install --quiet --upgrade pip && .venv/bin/python -m pip install --quiet -r requirements.txt; fi
fi
exec .venv/bin/python -m streamlit run dashboard/app.py "$@"
