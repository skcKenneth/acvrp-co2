@echo off
REM ============================================================================
REM  check_status.bat
REM  ----------------------------------------------------------------------------
REM  Quick "did my overnight run succeed?" checker.
REM
REM  Looks at the most recent overnight_*.log and prints a summary of which
REM  steps finished, plus any [FATAL] / [WARNING] markers.
REM
REM  Usage:  check_status.bat
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist logs (
    echo No logs/ directory yet -- the overnight pipeline has not been run.
    exit /b 1
)

REM Find the latest log
set LATEST=
for /f "delims=" %%f in ('dir /b /od /a-d logs\overnight_*.log 2^>nul') do set LATEST=%%f

if "!LATEST!"=="" (
    echo No overnight_*.log found in logs\.
    exit /b 1
)

echo Latest log: logs\!LATEST!
echo.
echo === Step status (greps) ===
findstr /n "Step.*OK\| failed\|FATAL\|WARNING\|Pipeline finished" "logs\!LATEST!"

echo.
echo === Tail of log (last 30 lines) ===
powershell -Command "Get-Content 'logs\!LATEST!' -Tail 30"

echo.
echo === Checkpoint files present ===
if exist models\matnet_cvrp_best.pt (echo   [OK] models\matnet_cvrp_best.pt) else (echo   [missing] models\matnet_cvrp_best.pt)
if exist models\baseline_am_best.pt  (echo   [OK] models\baseline_am_best.pt) else (echo   [missing] models\baseline_am_best.pt)

echo.
echo === Results files present ===
if exist results\summary.csv         (echo   [OK] results\summary.csv          ^(Stage 1^)) else (echo   [missing] results\summary.csv)
if exist results\routes_map.html     (echo   [OK] results\routes_map.html      ^(Stage 1 map^)) else (echo   [missing] results\routes_map.html)
if exist results\summary_full.csv    (echo   [OK] results\summary_full.csv     ^(Stage 4 grid^)) else (echo   [missing] results\summary_full.csv)

endlocal
