@echo off
setlocal
rem ---------------------------------------------------------------------------
rem  Fetch the public datasets and process them into the eight Roche66k subsets.
rem
rem  Double-click this file, read the licence table, type  I AGREE , optionally
rem  answer  yes  to fetch rxn_yields from its mirror, then press Enter to
rem  process.  Close the window at the second prompt to stop without processing.
rem
rem  Files are written straight into  data\ .  Note that the upstream
rem  bh-hte-ood table is the full 357k-row release while the packaged copy
rem  is the 27,497-row subset: check before replacing an existing file.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

%PY% scripts\fetch_datasets.py --interactive %*
if errorlevel 1 goto :fail

echo.
echo  Finished.  The files are in data\
if not defined DSH_NO_PAUSE pause
exit /b 0

:fail
echo.
echo  Something went wrong -- see the messages above.
if not defined DSH_NO_PAUSE pause
exit /b 1
