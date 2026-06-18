@echo off
echo ====================================================================
echo Starting Amazon Sentiment Analytics & Diagnostics Web Application...
echo ====================================================================

REM Navigate to workspace base directory
cd /d "%~dp0"

REM Check if python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Verify models exist, train if not present
if not exist "models\model.pkl" (
    echo [INFO] Serialized model not found. Training model first...
    python src\train.py
    if %errorlevel% neq 0 (
        echo [ERROR] Model training failed. Check logs.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Verified model.pkl and vectorizer.pkl.
)

REM Start FastAPI application
echo [INFO] Initializing FastAPI server...
echo [INFO] Dashboard URL: http://127.0.0.1:8001
python -m uvicorn src.app:app --host 127.0.0.1 --port 8001

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start uvicorn. Please install requirements: pip install -r requirements.txt
    pause
)
