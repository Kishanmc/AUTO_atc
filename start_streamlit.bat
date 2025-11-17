@echo off
echo ========================================
echo Starting AutoATC Streamlit Frontend
echo ========================================
echo.

cd /d "%~dp0frontend\streamlit_app"

echo Checking dependencies...
python -m pip install --quiet streamlit requests pillow pandas 2>nul

echo.
echo Starting Streamlit app...
echo Frontend will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python -m streamlit run app.py

pause

