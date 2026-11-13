@echo off
setlocal
if /I not "%~1"=="install" (
    echo Usage: install.bat install
    exit /b 1
)
if not defined PYTHON_PATH (
    echo ERROR: PYTHON_PATH is not defined. Set it to the isolated Python runtime directory.
    exit /b 1
)
if not exist "%PYTHON_PATH%\python.exe" (
    echo ERROR: Could not find "%PYTHON_PATH%\python.exe"
    exit /b 1
)
"%PYTHON_PATH%\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo Dependency installation failed.
    exit /b 1
)
echo Dependencies installed.
exit /b 0
