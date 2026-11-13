@echo off
REM Menjalankan aplikasi prediksi pembatalan pesanan
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Membuat virtual environment...
    python -m venv .venv
    echo [2/2] Memasang dependensi...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
    echo Virtual environment ditemukan.
)

echo.
echo Membuka aplikasi di browser...
".venv\Scripts\python.exe" -m streamlit run app.py

pause
