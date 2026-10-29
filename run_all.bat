@echo off
setlocal
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

rem --- preflight 0: environment.  Missing packages used to surface as a bare
rem     ModuleNotFoundError several minutes into a step; check them first.
%PY% scripts\check_env.py
if errorlevel 1 goto :fail

rem --- preflight 1: the writers create these, but create them here too so a first
rem     run never dies on FileNotFoundError.
if not exist results mkdir results
if not exist outputs mkdir outputs
if not exist figures mkdir figures

rem --- preflight: the datasets themselves are not shipped; fetch them first ------
if not exist "data\(AI_RCP_pub)bh_data_clean_all_whitelisted.csv" (
  echo.
  echo  The datasets are not in data\ yet.
  echo.
  echo  Order of operations on a fresh checkout:
  echo    [1] python scripts\check_env.py --strict
  echo    [2] fetch_datasets.bat  -  answer I AGREE; rxn_yields asks a second time: yes
  echo    [3] run_all.bat  -  this file
  echo.
  goto :fail
)

rem --- machine-wide lock (opt-in).  The CPU-only xgboost wheel made concurrent runs safe,
rem     so the lock is OFF unless CD_LOCK=1 is set in the environment.
if not "%CD_LOCK%"=="1" goto :lock_off
%PY% scripts\pipeline_lock.py acquire --kind run_all --ttl-minutes 360
if errorlevel 3 goto :busy
if errorlevel 1 goto :fail
goto :lock_ready
:lock_off
echo  (run lock disabled - set CD_LOCK=1 to enable)
:lock_ready

echo ============================================================
echo  Step 1/12  grid, ceilings and Table 1 from data\*.csv
echo ============================================================
%PY% scripts\build_grid_ceiling.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 2/12  section 3.1 statistics
echo ============================================================
%PY% scripts\build_grid_stats.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 3/12  cross-fitted enrichment audit
echo ============================================================
%PY% scripts\build_enrichment.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 4/12  cross-dataset transfer and the Figure 5 view
echo ============================================================
%PY% scripts\build_transfer.py
if errorlevel 1 goto :fail
%PY% scripts\build_transfer_check.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 5/12  the three validation protocols
echo ============================================================
%PY% scripts\build_protocols.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 6/12  structural baselines (Table 2)
echo ============================================================
%PY% scripts\build_table2.py --arms 1 2 3 3b 5
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 7/12  section 3.6: held-out pairs and unseen reactants
echo ============================================================
%PY% scripts\build_novel_pairs.py
if errorlevel 1 goto :fail
%PY% scripts\build_unseen_reactants.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 8/12  virtual library: mu/sigma, BO panels, recommendation table
echo ============================================================
%PY% scripts\build_acquisition.py
if errorlevel 1 goto :fail
%PY% scripts\build_bo_panel.py
if errorlevel 1 goto :fail
%PY% scripts\build_heldout_acquisition.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 9/12  attribution tables (one subprocess per dataset, ~20 min)
echo ============================================================
%PY% scripts\build_attrib13.py --tag A --per-dataset
if errorlevel 1 goto :fail
%PY% scripts\build_attrib_postprocess.py
if errorlevel 1 goto :fail
%PY% scripts\build_attrib_flags.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 10/12  render the paper figures
echo ============================================================
%PY% scripts\figures_paper2.py
if errorlevel 1 goto :fail
%PY% scripts\figures.py --only all --allow-heavy
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 11/12  verify every number quoted in the manuscript
echo ============================================================
rem the gate reads results\bootstrap_ci.csv; it was never produced by this script
%PY% scripts\bootstrap_ci.py
if errorlevel 1 goto :fail

%PY% scripts\build_final_numbers.py

if errorlevel 1 goto :fail

%PY% scripts\verify_numbers.py
if errorlevel 1 goto :fail

echo.
call :release_lock
echo ALL DONE - tables in results\, figures in figures\
if not defined DSH_NO_PAUSE pause
exit /b 0

:busy
echo.
echo  Another pipeline is already running on this machine.
echo  Wait for it to finish, then start this one again.
if not defined DSH_NO_PAUSE pause
exit /b 3

:release_lock
if not "%CD_LOCK%"=="1" exit /b 0
%PY% scripts\pipeline_lock.py release
exit /b 0

:fail
echo.
echo FAILED - see the messages above
if not defined DSH_NO_PAUSE pause
exit /b 1
