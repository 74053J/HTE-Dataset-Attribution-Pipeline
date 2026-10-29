@echo off
setlocal
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

rem --- preflight: the datasets themselves are not shipped; fetch them first ------
if not exist "data\(AI_RCP_pub)bh_data_clean_all_whitelisted.csv" (
  echo.
  echo  The datasets are not in data\ yet.
  echo  Run  fetch_datasets.bat  once to download and prepare them,
  echo  then run this file again.
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
echo  Step 1/10  grid, ceilings and Table 1 from data\*.csv
echo ============================================================
%PY% scripts\build_grid_ceiling.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 2/10  section 3.1 statistics
echo ============================================================
%PY% scripts\build_grid_stats.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 3/10  cross-fitted enrichment audit
echo ============================================================
%PY% scripts\build_enrichment.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 4/10  cross-dataset transfer and the Figure 5 view
echo ============================================================
%PY% scripts\build_transfer.py
if errorlevel 1 goto :fail
%PY% scripts\build_transfer_check.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 5/10  the three validation protocols
echo ============================================================
%PY% scripts\build_protocols.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 6/10  structural baselines (Table 2)
echo ============================================================
%PY% scripts\build_table2.py --arms 1 2 3 3b 5
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 7/10  section 3.6: held-out pairs and unseen reactants
echo ============================================================
%PY% scripts\build_novel_pairs.py
if errorlevel 1 goto :fail
%PY% scripts\build_unseen_reactants.py
if errorlevel 1 goto :fail
%PY% scripts\build_heldout_acquisition.py

if errorlevel 1 goto :fail


echo ============================================================
echo  Step 8/10  attribution tables (about 21 minutes)
echo ============================================================
%PY% scripts\build_attrib13.py --tag A
if errorlevel 1 goto :fail
%PY% scripts\build_attrib_postprocess.py
if errorlevel 1 goto :fail
%PY% scripts\build_attrib_flags.py
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 9/10  render the paper figures (virtual library skipped)
echo ============================================================
%PY% scripts\figures_paper2.py
if errorlevel 1 goto :fail
%PY% scripts\figures.py --only all
if errorlevel 1 goto :fail

echo ============================================================
echo  Step 10/10  verify every number quoted in the manuscript
echo ============================================================
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
