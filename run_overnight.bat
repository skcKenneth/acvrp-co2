@echo off
REM ============================================================================
REM  run_overnight.bat  (v3 -- defensive checkpoint handling)
REM  ----------------------------------------------------------------------------
REM  Unattended end-to-end pipeline for the ACVRP-CO2 project.
REM
REM  Major changes vs v2:
REM    - Detects actual checkpoint filenames (handles both old and new naming).
REM    - Verifies the .pt file exists before claiming success.
REM    - Prints a clear summary table at the end.
REM    - Step 5 / 5b explicitly check that NCO models loaded successfully.
REM
REM  Optional flags:
REM    /summary-only   Skip all steps; only run the final summary block
REM                    (for testing batch parsing after an overnight run).
REM    /shutdown       Power off 60 s after the pipeline completes.
REM ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

if /i "%~1"=="/summary-only" goto summary_only
goto start_full_pipeline

REM ===========================================================================
REM /summary-only  -- test the tail without re-running the pipeline
REM ===========================================================================
:summary_only
for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set STAMP=%%t
if "!STAMP!"=="" set STAMP=run
if not exist logs mkdir logs
set LOG=logs\summary_only_!STAMP!.log
echo. > "%LOG%"
echo Summary-only test >> "%LOG%"
echo Started: %date% %time% >> "%LOG%"
echo. >> "%LOG%"

set MACAU_S1_OK=0
if exist results_macau\summary.csv set MACAU_S1_OK=1
set HK_S1_OK=0
if exist results_hongkong\summary.csv set HK_S1_OK=1
set MATNET_TRAIN_OK=0
if exist models\matnet_cvrp_n20_best.pt set MATNET_TRAIN_OK=1
if "!MATNET_TRAIN_OK!"=="0" if exist models\matnet_cvrp_best.pt set MATNET_TRAIN_OK=1
set BASELINE_TRAIN_OK=0
if exist models\baseline_am_n20_best.pt set BASELINE_TRAIN_OK=1
if "!BASELINE_TRAIN_OK!"=="0" if exist models\baseline_am_best.pt set BASELINE_TRAIN_OK=1
set MACAU_S5_OK=0
if exist results_macau\summary_full.csv set MACAU_S5_OK=1
set HK_S5_OK=0
if exist results_hongkong\summary_full.csv set HK_S5_OK=1

echo Summary-only mode: checking outputs on disk, no Python steps.
echo Logging to %LOG%
echo.
goto write_summary

:start_full_pipeline
REM --- Time-stamp via PowerShell (full pipeline) ---
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

REM ===========================================================================
REM Step 0: environment sanity
REM ===========================================================================
echo. >> "%LOG%"
echo [%time%] Step 0: Environment sanity check... >> "%LOG%"
echo [%time%] Step 0: Environment sanity check...
python -c "import torch, ortools, osmnx, deap, yaml; print('All imports OK'); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); assert torch.cuda.is_available(), 'CUDA not available'" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FATAL] Step 0 failed. >> "%LOG%"
    echo FATAL: environment check failed. Open %LOG% for details.
    exit /b 1
)

REM ===========================================================================
REM Step 1: classical Stage 1 on Macau
REM ===========================================================================
echo. >> "%LOG%"
echo [%time%] Step 1: Classical experiment (Macau)... >> "%LOG%"
echo [%time%] Step 1: Macau classical experiment...
python -m src.experiments --config config.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FATAL] Step 1 failed. Pipeline aborted. >> "%LOG%"
    echo FATAL: classical Stage 1 Macau failed. Open %LOG%.
    exit /b 2
)
set MACAU_S1_OK=1
echo [%time%] Step 1 OK. >> "%LOG%"

REM ===========================================================================
REM Step 1b: classical Stage 1 on Hong Kong (non-fatal)
REM ===========================================================================
set HK_S1_OK=0
if exist config_hongkong.yaml if exist data\customers_hongkong.csv (
    echo. >> "%LOG%"
    echo [%time%] Step 1b: Classical experiment Hong Kong... >> "%LOG%"
    echo [%time%] Step 1b: Hong Kong classical experiment...
    python -m src.experiments --config config_hongkong.yaml >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [WARNING] Step 1b failed; continuing. >> "%LOG%"
    ) else (
        echo [%time%] Step 1b OK. >> "%LOG%"
        set HK_S1_OK=1
    )
) else (
    echo [%time%] Step 1b skipped: HK config or data missing. >> "%LOG%"
    echo [%time%] Step 1b skipped.
)

REM ===========================================================================
REM Step 2: NCO smoke test (fatal if it fails)
REM ===========================================================================
echo. >> "%LOG%"
echo [%time%] Step 2: NCO smoke test... >> "%LOG%"
echo [%time%] Step 2: NCO smoke test...
python -m src.nco_experiments --mode train --config configs/nco_config_smoke.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FATAL] Step 2 smoke test failed. >> "%LOG%"
    echo FATAL: NCO smoke test failed. Open %LOG%.
    exit /b 3
)
echo [%time%] Step 2 OK. >> "%LOG%"

REM ===========================================================================
REM Step 3: Train MatNet-CVRP (N=20)
REM ===========================================================================
echo. >> "%LOG%"
echo [%time%] Step 3: Training MatNet-CVRP (N=20)... >> "%LOG%"
echo [%time%] Step 3: Training MatNet-CVRP (~60-90 min)...
python -m src.train_nco --policy matnet --config configs/train.yaml --osm-eval >> "%LOG%" 2>&1
set MATNET_TRAIN_OK=0
if not errorlevel 1 set MATNET_TRAIN_OK=1

REM Find the actual checkpoint that was produced. New code: matnet_cvrp_n20_best.pt;
REM old code: matnet_cvrp_best.pt. Detect both.
set MATNET_CKPT=
if exist "models\matnet_cvrp_n20_best.pt" set MATNET_CKPT=models\matnet_cvrp_n20_best.pt
if "!MATNET_CKPT!"=="" if exist "models\matnet_cvrp_best.pt" set MATNET_CKPT=models\matnet_cvrp_best.pt

if "!MATNET_CKPT!"=="" (
    echo [WARNING] MatNet training: no checkpoint file detected. >> "%LOG%"
    set MATNET_TRAIN_OK=0
) else (
    echo [%time%] MatNet checkpoint: !MATNET_CKPT! >> "%LOG%"
    echo [%time%] MatNet checkpoint: !MATNET_CKPT!
)

REM ===========================================================================
REM Step 4: Train Vanilla-AM baseline (N=20)
REM ===========================================================================
echo. >> "%LOG%"
echo [%time%] Step 4: Training Vanilla-AM (N=20)... >> "%LOG%"
echo [%time%] Step 4: Training Vanilla-AM (~60-80 min)...
python -m src.train_nco --policy baseline --config configs/train.yaml --osm-eval >> "%LOG%" 2>&1
set BASELINE_TRAIN_OK=0
if not errorlevel 1 set BASELINE_TRAIN_OK=1

set BASELINE_CKPT=
if exist "models\baseline_am_n20_best.pt" set BASELINE_CKPT=models\baseline_am_n20_best.pt
if "!BASELINE_CKPT!"=="" if exist "models\baseline_am_best.pt" set BASELINE_CKPT=models\baseline_am_best.pt

if "!BASELINE_CKPT!"=="" (
    echo [WARNING] Baseline training: no checkpoint file detected. >> "%LOG%"
    set BASELINE_TRAIN_OK=0
) else (
    echo [%time%] Baseline checkpoint: !BASELINE_CKPT! >> "%LOG%"
    echo [%time%] Baseline checkpoint: !BASELINE_CKPT!
)

REM ===========================================================================
REM Step 5: grid comparison on Macau
REM ===========================================================================
echo. >> "%LOG%"
echo [%time%] Step 5: Grid comparison on Macau... >> "%LOG%"
echo [%time%] Step 5: Grid comparison on Macau...

set GRID_CMD=python -m src.experiments_full --config config.yaml
if not "!MATNET_CKPT!"=="" set GRID_CMD=!GRID_CMD! --matnet-checkpoint !MATNET_CKPT!
if not "!BASELINE_CKPT!"=="" set GRID_CMD=!GRID_CMD! --baseline-checkpoint !BASELINE_CKPT!

echo Command: !GRID_CMD! >> "%LOG%"
!GRID_CMD! >> "%LOG%" 2>&1
set MACAU_S5_OK=0
if not errorlevel 1 set MACAU_S5_OK=1
if "!MACAU_S5_OK!"=="1" (
    echo [%time%] Step 5 OK. >> "%LOG%"
) else (
    echo [WARNING] Step 5 Macau grid produced an error. >> "%LOG%"
)

REM ===========================================================================
REM Step 5b: grid comparison on Hong Kong
REM ===========================================================================
set HK_S5_OK=0
if exist config_hongkong.yaml if exist data\customers_hongkong.csv (
    echo. >> "%LOG%"
    echo [%time%] Step 5b: Grid comparison on Hong Kong... >> "%LOG%"
    echo [%time%] Step 5b: Grid comparison on Hong Kong...

    set GRID_CMD=python -m src.experiments_full --config config_hongkong.yaml
    if not "!MATNET_CKPT!"=="" set GRID_CMD=!GRID_CMD! --matnet-checkpoint !MATNET_CKPT!
    if not "!BASELINE_CKPT!"=="" set GRID_CMD=!GRID_CMD! --baseline-checkpoint !BASELINE_CKPT!

    echo Command: !GRID_CMD! >> "%LOG%"
    !GRID_CMD! >> "%LOG%" 2>&1
    if not errorlevel 1 set HK_S5_OK=1
    if "!HK_S5_OK!"=="1" (
        echo [%time%] Step 5b OK. >> "%LOG%"
    ) else (
        echo [WARNING] Step 5b produced an error. >> "%LOG%"
    )
) else (
    echo [%time%] Step 5b skipped: HK config or data missing. >> "%LOG%"
)

goto write_summary

REM ===========================================================================
REM Summary (shared by full pipeline and /summary-only)
REM ===========================================================================
:write_summary
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo Pipeline finished: %date% %time% >> "%LOG%"
echo. >> "%LOG%"
echo Step results: >> "%LOG%"
call :label_ok "  Step 1  Macau classical" !MACAU_S1_OK!
call :label_ok "  Step 1b HK classical" !HK_S1_OK!
call :label_ok "  Step 3  MatNet training" !MATNET_TRAIN_OK!
call :label_ok "  Step 4  Vanilla training" !BASELINE_TRAIN_OK!
call :label_ok "  Step 5  Macau grid" !MACAU_S5_OK!
call :label_ok "  Step 5b HK grid" !HK_S5_OK!
echo. >> "%LOG%"
echo Result files: >> "%LOG%"
call :check_file "results_macau\summary.csv"
call :check_file "results_macau\summary_full.csv"
call :check_file "results_hongkong\summary.csv"
call :check_file "results_hongkong\summary_full.csv"
echo ============================================================ >> "%LOG%"

echo.
if /i "%~1"=="/summary-only" (
    echo Summary-only test finished. See %LOG%
) else (
    echo Pipeline finished. See %LOG% for the full log.
    echo Run check_status.bat for a summary.
)

REM ===========================================================================
REM Optional shutdown
REM ===========================================================================
if /i "%~1"=="/summary-only" goto summary_only_exit
if /i "%~1"=="/shutdown" (
    echo Shutdown requested. PC will power off in 60 seconds.
    echo To cancel: open a new cmd and run  shutdown /a
    shutdown /s /t 60 /c "ACVRP-CO2 overnight pipeline complete."
)

:summary_only_exit
endlocal
exit /b 0


REM ===========================================================================
REM Subroutines
REM ===========================================================================

:label_ok
if "%~2"=="1" (
    echo %~1 OK >> "%LOG%"
) else (
    echo %~1 FAIL-or-SKIP >> "%LOG%"
)
exit /b 0

:check_file
if exist "%~1" (
    echo   %~1 present >> "%LOG%"
) else (
    echo   %~1 missing >> "%LOG%"
)
exit /b 0
