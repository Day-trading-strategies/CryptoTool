@echo off
echo ========================================
echo    Crypto Price Monitor Setup
echo ========================================
echo.

echo [1/4] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/4] Setup complete!
echo.
echo ========================================
echo    Setup Complete! 
echo ========================================
echo.
echo To run the app:
echo   1. Double-click 'run_app.bat'
echo   2. Or run: streamlit run main.py
echo.
echo The app will open at: http://localhost:8501
echo.
pause
