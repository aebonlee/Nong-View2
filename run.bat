@echo off
REM Nong-View2 Pipeline Batch Runner for Windows

echo ===============================================
echo Nong-View2 Agricultural AI Analysis Pipeline
echo ===============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check command line arguments
if "%1"=="" (
    echo.
    echo Usage: run.bat [command] [options]
    echo.
    echo Commands:
    echo   full      - Run full pipeline
    echo   test      - Run tests
    echo   example   - Run example script
    echo   help      - Show help
    echo.
    goto end
)

REM Execute commands
if "%1"=="full" (
    echo Running full pipeline...
    python main.py %2 %3 %4 %5 %6 %7 %8 %9
    goto end
)

if "%1"=="test" (
    echo Running tests...
    python -m pytest tests/ -v
    if %errorlevel% neq 0 (
        python tests/test_pipeline.py
    )
    goto end
)

if "%1"=="example" (
    echo Running example...
    python run_example.py
    goto end
)

if "%1"=="help" (
    python main.py --help
    goto end
)

echo Unknown command: %1
echo Run 'run.bat' without arguments for help

:end
echo.
echo ===============================================
echo Process completed
echo ===============================================
pause