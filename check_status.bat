@echo off
REM ============================================================================
REM  check_status.bat
REM  ----------------------------------------------------------------------------
REM  Quick "did my overnight run succeed?" checker for the Macau + Hong Kong
REM  pipeline.
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
REM Check both old (no suffix) and new (n20/n50) naming conventions
set FOUND_MATNET=0
if exist models\matnet_cvrp_n20_best.pt (echo   [OK]      models\matnet_cvrp_n20_best.pt & set FOUND_MATNET=1)
if exist models\matnet_cvrp_n50_best.pt (echo   [OK]      models\matnet_cvrp_n50_best.pt & set FOUND_MATNET=1)
if exist models\matnet_cvrp_best.pt     (echo   [legacy]  models\matnet_cvrp_best.pt     ^(rename to add _n20 or _n50 suffix^) & set FOUND_MATNET=1)
if "!FOUND_MATNET!"=="0" echo   [missing] no matnet_cvrp_*_best.pt found

set FOUND_BASE=0
if exist models\baseline_am_n20_best.pt (echo   [OK]      models\baseline_am_n20_best.pt & set FOUND_BASE=1)
if exist models\baseline_am_n50_best.pt (echo   [OK]      models\baseline_am_n50_best.pt & set FOUND_BASE=1)
if exist models\baseline_am_best.pt     (echo   [legacy]  models\baseline_am_best.pt     ^(rename to add _n20 or _n50 suffix^) & set FOUND_BASE=1)
if "!FOUND_BASE!"=="0" echo   [missing] no baseline_am_*_best.pt found

echo.
echo === Macau results ===
if exist results_macau\summary.csv         (echo   [OK] results_macau\summary.csv          ^(Stage 1^)) else (echo   [missing] results_macau\summary.csv)
if exist results_macau\routes_map.html     (echo   [OK] results_macau\routes_map.html      ^(Stage 1 map^)) else (echo   [missing] results_macau\routes_map.html)
if exist results_macau\summary_full.csv    (echo   [OK] results_macau\summary_full.csv     ^(Stage 5 grid^)) else (echo   [missing] results_macau\summary_full.csv)

echo.
echo === Hong Kong results ===
if exist results_hongkong\summary.csv      (echo   [OK] results_hongkong\summary.csv       ^(Stage 1b^)) else (echo   [missing] results_hongkong\summary.csv)
if exist results_hongkong\routes_map.html  (echo   [OK] results_hongkong\routes_map.html   ^(Stage 1b map^)) else (echo   [missing] results_hongkong\routes_map.html)
if exist results_hongkong\summary_full.csv (echo   [OK] results_hongkong\summary_full.csv  ^(Stage 5b grid^)) else (echo   [missing] results_hongkong\summary_full.csv)

endlocal
