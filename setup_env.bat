@echo off
rem ============================================================================
rem  setup_env.bat  --  one-click environment for HTE-Dataset-Attribution-Pipeline
rem
rem  What it does
rem    1. finds a Python 3.9-3.12 interpreter (py launcher first, then python)
rem    2. creates .venv\ here if it does not exist yet
rem    3. upgrades pip / setuptools / wheel
rem    4. installs requirements.txt, then the CPU build of torch
rem    5. runs scripts\check_env.py and lists whatever is still missing
rem
rem  Network policy  (no proxy is assumed -- most machines do not have one)
rem    * default is a DIRECT connection to PyPI, with a short timeout;
rem    * the link is probed first.  If it fails, times out, or averages below
rem      SPEED_MIN KB/s, the run switches to the Tsinghua mirror and tries again;
rem    * every pip call also has a second attempt on the other index.
rem
rem  Usage
rem    setup_env.bat            install (direct, mirror on failure or slow link)
rem    setup_env.bat check      report what would happen, install nothing
rem    setup_env.bat mirror     install from the Tsinghua mirror straight away
rem    setup_env.bat proxy      install through a local http://127.0.0.1:7897
rem ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "MODE=%~1"
set "FAILED=0"
set "PROXY=http://127.0.0.1:7897"
set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
set "SPEED_MIN=60"

rem --- default: no proxy.  Clear anything inherited, including the scheme-less
rem     registry entry that makes pip die with "check_hostname requires server_hostname".
if /i "%MODE%"=="proxy" (
  set "HTTP_PROXY=%PROXY%"
  set "HTTPS_PROXY=%PROXY%"
  set "http_proxy=%PROXY%"
  set "https_proxy=%PROXY%"
  set "NO_PROXY=localhost,127.0.0.1,::1"
  set "no_proxy=localhost,127.0.0.1,::1"
) else (
  set "HTTP_PROXY="
  set "HTTPS_PROXY="
  set "http_proxy="
  set "https_proxy="
  set "NO_PROXY=*"
  set "no_proxy=*"
)

echo.
echo ============================================================
echo  HTE-Dataset-Attribution-Pipeline - environment setup
echo  folder: %CD%
echo ============================================================
echo.

rem ---------------------------------------------------------------- 1. python
set "BASE_PY="
for %%V in (3.10 3.11 3.12 3.9) do (
  if not defined BASE_PY (
    py -%%V -c "import sys" >nul 2>&1 && set "BASE_PY=py -%%V"
  )
)
if not defined BASE_PY (
  python -c "import sys" >nul 2>&1 && set "BASE_PY=python"
)
if not defined BASE_PY (
  echo [ERROR] No Python found.  Install Python 3.10 ^(64-bit^) from python.org,
  echo         tick "Add python.exe to PATH", then run this file again.
  goto :end
)

for /f "delims=" %%V in ('%BASE_PY% -c "import sys;print(sys.version.split()[0])" 2^>nul') do set "PYVER=%%V"
for /f "delims=" %%P in ('%BASE_PY% -c "import sys;print(sys.executable)" 2^>nul') do set "PYPATH=%%P"
echo [1/5] Python %PYVER%
echo        %PYPATH%

if exist ".venv\Scripts\python.exe" (
  echo [2/5] .venv already present - reusing it
) else (
  echo [2/5] creating .venv ...
  if /i "%MODE%"=="check" (
    echo        ^(check mode: skipped^)
  ) else (
    %BASE_PY% -m venv .venv
    if errorlevel 1 (
      echo [ERROR] python -m venv failed.  On Debian/Ubuntu run:
      echo         sudo apt install python3-venv
      goto :end
    )
  )
)

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=%PYPATH%"
echo        interpreter: %VPY%

rem ------------------------------------------------------- 2. choose an index
set "INDEX_FLAG="
set "INDEX_NAME=direct PyPI"
if /i "%MODE%"=="mirror" (
  set "INDEX_FLAG=--index-url %MIRROR%"
  set "INDEX_NAME=Tsinghua mirror"
)
if /i not "%MODE%"=="check" if /i not "%MODE%"=="mirror" (
  echo [3/5] probing PyPI ^(direct, %SPEED_MIN% KB/s required^) ...
  set "PROBE_OUT=%TEMP%\hte_net_probe.txt"
  if exist "%PROBE_OUT%" del "%PROBE_OUT%" >nul 2>&1
  for /f "delims=" %%R in ('""%VPY%" "scripts\_net_probe.py" --decide %SPEED_MIN% --out "%PROBE_OUT%" 2^>nul"') do echo        %%R
  set "INDEX_FLAG="
  if exist "%PROBE_OUT%" for /f "usebackq delims=" %%R in ("%PROBE_OUT%") do set "INDEX_FLAG=%%R"
  if defined INDEX_FLAG (
    set "INDEX_NAME=Tsinghua mirror"
  ) else (
    set "INDEX_NAME=direct PyPI"
  )
  echo        using: %INDEX_NAME%
)

rem ---------------------------------------------------------------- 3. install
echo [4/5] installing pip/setuptools/wheel, requirements.txt, torch ^(cpu^)
if /i "%MODE%"=="check" (
  echo        ^(check mode: skipped^)
) else (
  call :piptry -m pip install --no-cache-dir --timeout 20 --retries 1 %INDEX_FLAG% --upgrade pip setuptools wheel
  if /i "%MODE%"=="notorch" (
    echo        torch: skipped ^(notorch^) - only the GNN arm of Step 6 needs it
  ) else (
    echo        torch: CPU build, installed ONCE ^(~120 MB^)
    call :torchtry
    if errorlevel 1 (
      echo        [WARN] CPU torch unavailable.  Everything still runs except the GNN
      echo               arm of Table 2 ^(Step 6, arm 5^).
    )
  )
  call :piptry -m pip install --no-cache-dir --timeout 20 --retries 1 %INDEX_FLAG% -r requirements.txt
  if errorlevel 1 set "FAILED=1"
)

rem ---------------------------------------------------------------- 4. verify
echo [5/5] checking the environment
if exist "%VPY%" (
  "%VPY%" scripts\check_env.py
  if errorlevel 1 (
    echo.
    echo [WARN] check_env reported missing pieces ^(see the list above^).
    echo        Re-run this file, or install the named packages by hand.
  )
) else (
  echo        ^(no interpreter to check with^)
)

echo.
if /i "%MODE%"=="check" (
  echo check mode finished - nothing was installed.
) else (
  echo ============================================================
  echo  Next:  fetch_datasets.bat   ^(answer I AGREE; rxn_yields asks again: yes^)
  echo         run_all.bat
  echo ============================================================
)
goto :end

rem --------------------------------------------------------------- helpers
:piptry
rem  %* = pip arguments.  Attempt 1 as configured; attempt 2 on the other index;
rem  attempt 3 through the local proxy, but only when MODE=proxy.
"%VPY%" %* && exit /b 0
echo        retry 1: switching index
set "ALT_FLAG=--index-url %MIRROR%"
if defined INDEX_FLAG set "ALT_FLAG="
"%VPY%" %* %ALT_FLAG% && exit /b 0
if /i "%MODE%"=="proxy" (
  echo        retry 2: through %PROXY%
  "%VPY%" %* --proxy %PROXY% && exit /b 0
)
echo        [FAIL] no index reachable.  Check the network and re-run this file,
echo               or try "setup_env.bat mirror" / "setup_env.bat proxy".
exit /b 1

:torchtry
rem  One shot at the CPU build; if the PyTorch CDN is unreachable, fall back to the
rem  selected PyPI index (its Windows wheel is a CUDA build -- bigger, but fast).
"%VPY%" -m pip install --no-cache-dir --timeout 30 --retries 1 torch --index-url https://download.pytorch.org/whl/cpu && exit /b 0
echo        CPU index unreachable - falling back to %INDEX_NAME%
"%VPY%" -m pip install --no-cache-dir --timeout 30 --retries 1 %INDEX_FLAG% torch && exit /b 0
exit /b 1

:end
rem pip extracts its bundled CA file to %TEMP%\tmp<random>cacert.pem once per run and never
rem removes it (pip._vendor.certifi.where -> importlib.resources).  Clean up our own mess.
del "%TEMP%\tmp*cacert.pem" >nul 2>&1
echo.
if /i not "%DSH_NO_PAUSE%"=="1" pause
endlocal
exit /b %FAILED%
