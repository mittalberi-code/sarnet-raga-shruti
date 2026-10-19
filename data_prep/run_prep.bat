@echo off
cd /d "%~dp0"
echo START %date% %time% > prep_log.txt
set PY=
where py >nul 2>&1 && set PY=py -3
if "%PY%"=="" (where python >nul 2>&1 && set PY=python)
if "%PY%"=="" (echo NO PYTHON FOUND >> prep_log.txt & exit /b 1)
echo using %PY% >> prep_log.txt
%PY% -c "import numpy" >nul 2>&1 || %PY% -m pip install --user numpy >> prep_log.txt 2>&1
%PY% prep_raga_features.py >> prep_log.txt 2>&1
echo END %date% %time% >> prep_log.txt
