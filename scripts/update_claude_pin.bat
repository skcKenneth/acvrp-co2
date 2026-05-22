@echo off
REM Regenerate docs/CLAUDE_PIN.md using Miniconda/Anaconda base environment
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
cd /d "%ROOT%"

set "ACTIVATE="
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not defined ACTIVATE if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" set "ACTIVATE=C:\ProgramData\miniconda3\Scripts\activate.bat"

if not defined ACTIVATE (
    echo [ERROR] Could not find Miniconda/Anaconda activate.bat
    echo Try: .\scripts\update_claude_pin.ps1
    exit /b 1
)

call "%ACTIVATE%" base
if errorlevel 1 (
    echo [ERROR] conda activate base failed
    exit /b 1
)

python "%SCRIPT_DIR%update_claude_pin.py" %*
exit /b %ERRORLEVEL%
