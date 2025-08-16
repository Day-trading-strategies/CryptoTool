@echo off
echo Starting Crypto Price Monitor...
echo.
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

REM Activate virtual environment and run the app
call .venv\Scripts\activate.bat
python -m streamlit run main.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ERROR: Something went wrong!
    echo Make sure you ran setup.bat first.
    echo.
    pause
)