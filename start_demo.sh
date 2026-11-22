#!/usr/bin/env bash
# Linux/Kaggle/Colab: menjalankan API (FastAPI :8000) + UI Streamlit (:8501)
cd "$(dirname "$0")"
PY=${PY:-python}
nohup $PY -m uvicorn tbp.serve:app --port 8000 > logs/serve_demo.log 2>&1 &
echo "Menunggu API siap..."
until curl -s localhost:8000/health >/dev/null; do sleep 2; done
nohup $PY -m streamlit run app_streamlit.py --server.port 8501 --server.headless true > logs/streamlit.log 2>&1 &
until curl -s localhost:8501/_stcore/health >/dev/null; do sleep 2; done
echo "API http://localhost:8000  |  UI http://localhost:8501"
