@echo off
REM ============================================================================
REM  run_overnight.bat
REM  ----------------------------------------------------------------------------
REM  Unattended end-to-end pipeline for the ACVRP-CO2 project.
REM
REM  Cities studied:
REM    Primary  : Macau Peninsula        (config.yaml)
REM    Secondary: Hong Kong Island core  (config_hongkong.yaml)
REM
REM  Pipeline:
REM    Step 0  : Verify PyTorch + CUDA
REM    Step 1  : Classical experiment on Macau   (~5 min CPU)
REM    Step 1b : Classical experiment on Hong Kong (~5 min CPU)
REM    Step 2  : NCO smoke test                  (~5 min GPU)
REM    Step 3  : Train MatNet-CVRP               (3-5 h on RTX 5070+)
REM    Step 4  : Train Vanilla-AM baseline       (2-3 h)
REM    Step 5  : 4-solver x 4-matrix grid on Macau
REM    Step 5b : Same grid on Hong Kong
REM
REM  Usage:
REM    run_overnight.bat              (no shutdown after)
REM    run_overnight.bat /shutdown    (auto-shutdown after run)
REM ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM --- Time-stamp via PowerShell (wmic was removed in Windows 11 25H2) ---
for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set STAMP=%%t
if "!STAMP!"=="" set STAMP=run

if not exist logs mkdir logs
set LOG=logs\overnight_!STAMP!.log

echo. > "%LOG%"
echo ============================================================ >> "%LOG%"
echo ACVRP-CO2 overnight pipeline (Macau + Hong Kong) >> "%LOG%"
echo Started: %date% %time% >> "%LOG%"
echo Working dir: %cd% >> "%LOG%"
echo Log file:    %LOG% >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo. >> "%LOG%"

echo Logging to %LOG%
echo.

REM ----------------------------------------------------------------------------
REM Step 0: environment sanity
REM ----------------------------------------------------------------------------
echo [%time%] Step 0: Environment sanity check... >> "%LOG%"
echo [%time%] Step 0: Environment sanity check...
python -c "import torch, ortools, osmnx, deap, yaml; print('All imports OK'); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); assert torch.cuda.is_available(), 'CUDA not available -- aborting overnight run.'" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FATAL] Step 0 failed. See "%LOG%" >> "%LOG%"
    echo FATAL: environment check failed. Open %LOG% for details.
    exit /b 1
)

REM ----------------------------------------------------------------------------
REM Step 1: classical Stage 1 on Macau
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 1: Classical experiment (Macau)... >> "%LOG%"
echo [%time%] Step 1: Macau classical experiment...
python -m src.experiments --config config.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FATAL] Step 1 failed. Pipeline aborted. >> "%LOG%"
    echo FATAL: classical Stage 1 Macau failed. Open %LOG%.
    exit /b 2
)
echo [%time%] Step 1 OK. >> "%LOG%"

REM ----------------------------------------------------------------------------
REM Step 1b: classical Stage 1 on Hong Kong
REM ----------------------------------------------------------------------------
if exist config_hongkong.yaml if exist data\customers_hongkong.csv (
    echo. >> "%LOG%"
    echo [%time%] Step 1b: Classical experiment Hong Kong... >> "%LOG%"
    echo [%time%] Step 1b: Hong Kong classical experiment...
    python -m src.experiments --config config_hongkong.yaml >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [WARNING] Step 1b failed; continuing with main pipeline. >> "%LOG%"
    ) else (
        echo [%time%] Step 1b OK. >> "%LOG%"
    )
) else (
    echo [%time%] Step 1b skipped: Hong Kong config or data missing. >> "%LOG%"
)

REM ----------------------------------------------------------------------------
REM Step 2: NCO smoke training
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 2: NCO smoke test... >> "%LOG%"
echo [%time%] Step 2: NCO smoke test...
python -m src.nco_experiments --mode train --config configs/nco_config_smoke.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FATAL] Step 2 smoke failed. Aborting before main training. >> "%LOG%"
    echo FATAL: NCO smoke test failed. Open %LOG%.
    exit /b 3
)
echo [%time%] Step 2 OK. >> "%LOG%"

REM ----------------------------------------------------------------------------
REM Step 3: Train MatNet-CVRP
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 3: Training MatNet-CVRP (N=50)... >> "%LOG%"
echo [%time%] Step 3: Training MatNet-CVRP (3-5 h)...
python -m src.train_nco --policy matnet --config configs/train_n50.yaml --osm-eval >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [WARNING] Step 3 failed. Continuing with baseline so you don't lose the night. >> "%LOG%"
    set MATNET_OK=0
) else (
    echo [%time%] Step 3 OK. >> "%LOG%"
    set MATNET_OK=1
)

REM ----------------------------------------------------------------------------
REM Step 4: Train Vanilla-AM baseline
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 4: Training Vanilla-AM baseline (N=50)... >> "%LOG%"
echo [%time%] Step 4: Training Vanilla-AM baseline (2-3 h)...
python -m src.train_nco --policy baseline --config configs/train_n50.yaml --osm-eval >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [WARNING] Step 4 failed. Skipping baseline in Stage 5 evaluation. >> "%LOG%"
    set BASELINE_OK=0
) else (
    echo [%time%] Step 4 OK. >> "%LOG%"
    set BASELINE_OK=1
)

REM ----------------------------------------------------------------------------
REM Step 5: 4-solver x 4-matrix comparison on Macau
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 5: 4-solver grid comparison on Macau... >> "%LOG%"
echo [%time%] Step 5: Grid comparison on Macau...

set GRID_CMD=python -m src.experiments_full --config config.yaml
if "!MATNET_OK!"=="1" set GRID_CMD=!GRID_CMD! --matnet-checkpoint models\matnet_cvrp_best.pt
if "!BASELINE_OK!"=="1" set GRID_CMD=!GRID_CMD! --baseline-checkpoint models\baseline_am_best.pt
!GRID_CMD! >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [WARNING] Step 5 Macau produced an error. >> "%LOG%"
) else (
    echo [%time%] Step 5 OK. >> "%LOG%"
)

REM ----------------------------------------------------------------------------
REM Step 5b: 4-solver x 4-matrix comparison on Hong Kong
REM ----------------------------------------------------------------------------
if exist config_hongkong.yaml if exist data\customers_hongkong.csv (
    echo. >> "%LOG%"
    echo [%time%] Step 5b: 4-solver grid comparison on Hong Kong... >> "%LOG%"
    echo [%time%] Step 5b: Grid comparison on Hong Kong...

    set GRID_CMD=python -m src.experiments_full --config config_hongkong.yaml
    if "!MATNET_OK!"=="1" set GRID_CMD=!GRID_CMD! --matnet-checkpoint models\matnet_cvrp_best.pt
    if "!BASELINE_OK!"=="1" set GRID_CMD=!GRID_CMD! --baseline-checkpoint models\baseline_am_best.pt
    !GRID_CMD! >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [WARNING] Step 5b Hong Kong produced an error. >> "%LOG%"
    ) else (
        echo [%time%] Step 5b OK. >> "%LOG%"
    )
) else (
    echo [%time%] Step 5b skipped: Hong Kong config or data missing. >> "%LOG%"
)

REM ----------------------------------------------------------------------------
REM Wrap-up
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo Pipeline finished: %date% %time% >> "%LOG%"
echo MATNET trained:  !MATNET_OK! >> "%LOG%"
echo BASELINE trained: !BASELINE_OK! >> "%LOG%"
echo Outputs in: results_macau\, results_hongkong\ >> "%LOG%"
echo Log: %LOG% >> "%LOG%"
echo ============================================================ >> "%LOG%"

echo.
echo Pipeline finished. See %LOG% for the full log.

REM ----------------------------------------------------------------------------
REM Optional shutdown
REM ----------------------------------------------------------------------------
if /i "%~1"=="/shutdown" (
    echo Shutdown requested. PC will power off in 60 seconds.
    echo To cancel: open a new cmd and run  shutdown /a
    shutdown /s /t 60 /c "ACVRP-CO2 overnight pipeline complete."
)

endlocal
exit /b 0
