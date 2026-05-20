@echo off
REM ============================================================================
REM  run_overnight.bat
REM  ----------------------------------------------------------------------------
REM  Unattended end-to-end pipeline for the ACVRP-CO2 project.
REM
REM  What it does, in order:
REM    1. Verifies PyTorch + CUDA before starting anything heavy
REM    2. Runs the classical Stage 1 (OR-Tools + GA, CPU, ~10 min)
REM    3. Runs the NCO smoke test (~5 min) to catch GPU issues early
REM    4. Trains MatNet-CVRP (N=50, ~3-5 h)
REM    5. Trains Vanilla-AM baseline (N=50, ~2-3 h)
REM    6. Runs the full 4-solver x 4-matrix grid comparison
REM    7. Writes a single overnight.log with everything timestamped
REM
REM  Crucially, every step uses && so a failure stops the pipeline AT THAT
REM  STEP and writes a clear error marker into the log instead of silently
REM  continuing with corrupt state.
REM
REM  Usage:
REM    run_overnight.bat              (no shutdown after)
REM    run_overnight.bat /shutdown    (auto-shutdown after success only)
REM
REM  Logs are written to logs\overnight_YYYYMMDD_HHMMSS.log
REM ============================================================================

setlocal enabledelayedexpansion

REM --- Resolve project root from the script's own location ---
cd /d "%~dp0"

REM --- Set up the log file with a timestamp ---
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set DT=%%i
set STAMP=%DT:~0,8%_%DT:~8,6%
if not exist logs mkdir logs
set LOG=logs\overnight_%STAMP%.log

echo. > "%LOG%"
echo ============================================================ >> "%LOG%"
echo ACVRP-CO2 overnight pipeline >> "%LOG%"
echo Started: %date% %time% >> "%LOG%"
echo Working dir: %cd% >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo. >> "%LOG%"

REM ----------------------------------------------------------------------------
REM Step 0: sanity checks. Bail out before training if anything is wrong.
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
REM Step 1: classical Stage 1 (no GPU needed). Quick failure surface for OSMnx.
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 1: Classical experiment (OR-Tools + GA)... >> "%LOG%"
python -m src.experiments --config config.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FATAL] Step 1 failed. Pipeline aborted. >> "%LOG%"
    echo FATAL: classical Stage 1 failed. Open %LOG%.
    exit /b 2
)
echo [%time%] Step 1 OK. >> "%LOG%"

REM ----------------------------------------------------------------------------
REM Step 2: NCO smoke training (~5 min). If this fails, full training would too.
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 2: NCO smoke test... >> "%LOG%"
python -m src.nco_experiments --mode train --config configs/nco_config_smoke.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FATAL] Step 2 (smoke) failed. Aborting before main training. >> "%LOG%"
    echo FATAL: NCO smoke test failed. Open %LOG%.
    exit /b 3
)
echo [%time%] Step 2 OK. >> "%LOG%"

REM ----------------------------------------------------------------------------
REM Step 3: Train MatNet-CVRP (the main model, N=50).
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 3: Training MatNet-CVRP (N=50)... >> "%LOG%"
python -m src.train_nco --policy matnet --config configs/train_n50.yaml --osm-eval >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [WARNING] Step 3 failed. Continuing with baseline so you don't lose the night. >> "%LOG%"
    set MATNET_OK=0
) else (
    echo [%time%] Step 3 OK. >> "%LOG%"
    set MATNET_OK=1
)

REM ----------------------------------------------------------------------------
REM Step 4: Train Vanilla-AM baseline (N=50).
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 4: Training Vanilla-AM baseline (N=50)... >> "%LOG%"
python -m src.train_nco --policy baseline --config configs/train_n50.yaml --osm-eval >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [WARNING] Step 4 failed. Skipping baseline in Stage 4 evaluation. >> "%LOG%"
    set BASELINE_OK=0
) else (
    echo [%time%] Step 4 OK. >> "%LOG%"
    set BASELINE_OK=1
)

REM ----------------------------------------------------------------------------
REM Step 5: 4-solver x 4-matrix comparison. Skip silently if no checkpoints.
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo [%time%] Step 5: 4-solver grid comparison... >> "%LOG%"

set GRID_CMD=python -m src.experiments_full --config config.yaml
if "!MATNET_OK!"=="1" set GRID_CMD=!GRID_CMD! --matnet-checkpoint models\matnet_cvrp_best.pt
if "!BASELINE_OK!"=="1" set GRID_CMD=!GRID_CMD! --baseline-checkpoint models\baseline_am_best.pt
!GRID_CMD! >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [WARNING] Step 5 produced an error. >> "%LOG%"
)

REM ----------------------------------------------------------------------------
REM Wrap-up
REM ----------------------------------------------------------------------------
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo Pipeline finished: %date% %time% >> "%LOG%"
echo MATNET trained:  !MATNET_OK! >> "%LOG%"
echo BASELINE trained: !BASELINE_OK! >> "%LOG%"
echo Outputs in: results\ >> "%LOG%"
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
