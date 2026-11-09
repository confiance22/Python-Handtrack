@echo off
REM Retrolens launcher - uses the project venv so 'python' does not need to be on PATH.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Create it first with one of:
    echo     py -m venv venv
    echo     "C:\Users\RSN\AppData\Local\Programs\Python\Python313\python.exe" -m venv venv
    echo Then install dependencies:
    echo     venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo [ERROR] App exited with an error. See the message above.
    pause
)
