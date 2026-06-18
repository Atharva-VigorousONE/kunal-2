@echo off
setlocal enabledelayedexpansion

echo ====================================================================
echo Setting up Amazon Sentiment Analytics ^& Diagnostics Web Application...
echo ====================================================================

REM Navigate to workspace base directory
cd /d "%~dp0"

REM 1. Check if python is installed
where python >nul 2>nul
if %errorlevel% equ 0 goto python_found
echo [ERROR] Python was not found in your system PATH.
echo Please install Python 3.8+ and try again.
pause
exit /b 1

:python_found
REM 2. Check Python version (>= 3.8)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"
if %errorlevel% equ 0 goto python_version_ok
echo [ERROR] Python version must be 3.8 or higher.
echo Current Python version is:
python --version
pause
exit /b 1

:python_version_ok
echo [INFO] Python version verified.

REM 3. Create virtual environment (.venv) if not already exists
if exist ".venv\Scripts\python.exe" goto venv_exists
echo [INFO] Creating virtual environment (.venv)...
python -m venv .venv
if %errorlevel% equ 0 goto venv_created
echo [ERROR] Failed to create virtual environment.
pause
exit /b 1

:venv_created
echo [INFO] Virtual environment created successfully.
goto venv_done

:venv_exists
echo [INFO] Virtual environment (.venv) already exists.

:venv_done
REM 4. Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% equ 0 goto venv_activated
echo [ERROR] Failed to activate virtual environment.
pause
exit /b 1

:venv_activated
REM 5. Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip. Continuing...
)

REM 6. Install requirements
echo [INFO] Installing required dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% equ 0 goto install_ok
echo [ERROR] Dependency installation failed.
pause
exit /b 1

:install_ok
echo [INFO] All dependencies installed successfully.

REM 7. Verify dataset path and run model training
if exist "archive (1)\7817_1.csv" goto dataset_ok
echo [WARNING] Dataset "archive (1)\7817_1.csv" was not found.
echo Please ensure the dataset is downloaded and extracted to: "%~dp0archive (1)\7817_1.csv"
goto setup_finish

:dataset_ok
echo [INFO] Dataset verified. Training the initial model...
python src\train.py
if %errorlevel% equ 0 goto train_ok
echo [ERROR] Model training failed. Check logs above.
pause
exit /b 1

:train_ok
echo [INFO] Model trained and metrics generated successfully.

:setup_finish
echo ====================================================================
echo Setup Completed Successfully!
echo ====================================================================
echo To start the web application, double-click run.bat or execute it in terminal.
echo ====================================================================
pause
