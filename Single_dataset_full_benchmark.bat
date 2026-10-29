@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem ---------------------------------------------------------------------------
rem  Run the frozen single-dataset pipeline on ONE dataset.
rem
rem  Double-click: you are asked for the CSV file name (Enter takes the default).
rem  Or pass it on the command line, quoted:
rem      Single_dataset_full_benchmark.bat "(Summit)Reizman_Suzuki_385.csv"
rem
rem  The file is looked up in . , data\ , Data\ , ..\data , ..\Data and next to
rem  the script.  Output goes to  Output_SingleDataset\<timestamp>_<dataset>\ .
rem
rem  NOTE: every value that can contain "(" or ")" is kept in a QUOTED variable and is
rem  never expanded inside an if-block -- an unquoted "(AI_RCP_pub)sm_....csv" made cmd
rem  abort on the spot, which is why the window could flash and close immediately.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

if not exist "archive\single_dataset_full_benchmark.py" goto :no_script

echo.
echo  Datasets available in data\ :
set FOUND=
for %%F in ("data\*.csv") do call :show "%%~nxF"
if not defined FOUND echo      (none - run fetch_datasets.bat first)
echo.

set "ARG=%~1"
if not defined ARG goto :ask

rem Feed the name through a temp file: a pipe would re-parse the "(" in the name.
set "NAMEFILE=%TEMP%\_ds_dataset_name.txt"
> "%NAMEFILE%" echo %ARG%
"%PY%" "archive\single_dataset_full_benchmark.py" < "%NAMEFILE%"
del "%NAMEFILE%" >nul 2>&1
goto :result

:ask
"%PY%" "archive\single_dataset_full_benchmark.py"

:result
if errorlevel 3 goto :busy
if errorlevel 1 goto :fail
echo.
echo  Done.  Results are in Output_SingleDataset\
set "RC=0"
goto :hold

:busy
echo.
echo  Another dataset is still running on this machine.
echo  Wait for it to finish, then run this file again.
set "RC=3"
goto :hold

:no_script
echo.
echo  archive\single_dataset_full_benchmark.py was not found.
echo  Run this file from the repository root (the folder that holds archive\ and data\).
set "RC=1"
goto :hold

:fail
echo.
echo  FAILED - see the messages above.
set "RC=1"

:hold
echo.
if not defined DSH_NO_PAUSE pause
set "RC=%RC%"
if not defined RC set "RC=0"
exit /b %RC%

:show
set "N=%~1"
echo !N! | findstr /I /B /C:"col_mappings" /C:"alias_mapping" >nul
if errorlevel 1 (
  echo      !N!
  set FOUND=1
)
exit /b 0
