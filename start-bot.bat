@echo off
setlocal

REM Define variable for Python in virtual environment
set "VENV_PYTHON=venv\Scripts\python.exe"

REM Check if virtual environment exists
if not exist %VENV_PYTHON% (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check for CUDA 11.8
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
if exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo CUDA 11.8 detected. Installing GPU requirements...
    %VENV_PYTHON% -m pip install --upgrade pip
    if %errorlevel% neq 0 (
        echo Error upgrading pip.
        pause
        exit /b %errorlevel%
    )
    pip install -r requirementsGPU.txt
    if %errorlevel% neq 0 (
        echo Error installing GPU dependencies.
        pause
        exit /b %errorlevel%
    )
) else (
    echo CUDA 11.8 not found. Installing CPU requirements...
    %VENV_PYTHON% -m pip install --upgrade pip
    if %errorlevel% neq 0 (
        echo Error upgrading pip.
        pause
        exit /b %errorlevel%
    )
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error installing CPU dependencies.
        pause
        exit /b %errorlevel%
    )
)

REM Run the main script
python main.py
if %errorlevel% neq 0 (
    echo Error running main.py.
    pause
    exit /b %errorlevel%
)

endlocal
pause