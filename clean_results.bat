@echo off
rem ---------------------------------------------------------------------------
rem  Clear everything run_all.bat generates, so a from-scratch reproduction is
rem  really from scratch.  results\legacy\ is a shipped archive, not generated.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
echo Clearing generated tables and figures ...
if exist results rmdir /s /q results\legacy_tmp 2>nul
for %%F in (results\*.csv) do del /q "%%F" >nul 2>&1
if exist figures for %%F in (figures\*.png) do del /q "%%F" >nul 2>&1
if exist outputs for %%F in (outputs\final_numbers\*.json) do del /q "%%F" >nul 2>&1
echo   results\*.csv, figures\*.png cleared (results\legacy\ kept)
echo Now run:  run_all.bat
if not defined DSH_NO_PAUSE pause
endlocal
