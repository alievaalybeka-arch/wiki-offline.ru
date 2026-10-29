@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Wiki Offline

where python >nul 2>nul
if errorlevel 1 (
    echo Python не найден. Установите Python 3.9+ с https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import libzim, numpy, onnxruntime, tokenizers" >nul 2>nul
if errorlevel 1 (
    echo Устанавливаю необходимые библиотеки...
    python -m pip install -r requirements.txt
)

python wiki_offline.py %*
pause
