@echo off
REM Semantic Shift Analyzer - Quick Start Script (Windows)
REM This script sets up and runs the application locally

echo.
echo ========================================
echo   Semantic Shift Analyzer - Quick Start
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo.

REM Check if Streamlit is installed
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies ^(this may take a few minutes^)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already installed
)

echo.
echo ========================================
echo   Setup complete!
echo ========================================
echo.
echo [INFO] Starting Streamlit app...
echo   The app will open in your browser automatically.
echo   If not, navigate to: http://localhost:8501
echo.
echo   Press Ctrl+C to stop the server
echo.

REM Run the app
streamlit run semantic_shift_app.py

pause
