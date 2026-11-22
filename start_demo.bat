@echo off
REM Menjalankan API (FastAPI, port 8000) dan UI Streamlit (port 8501)
cd /d %~dp0
start "TBP API" .venv\Scripts\python -m uvicorn tbp.serve:app --port 8000
echo Menunggu API siap...
timeout /t 8 /nobreak >nul
.venv\Scripts\python -m streamlit run app_streamlit.py --server.port 8501
