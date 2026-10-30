@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set LOG=e2e_result.txt

rem Always ask for the key (never reuse one that happens to be set in Windows), unless a .env file exists.
if not exist .env (
  set AGENTX_LLM_API_KEY=
  set /p AGENTX_LLM_API_KEY=Type the key for your local LLM gateway and press Enter: 
)

echo ===== AgentX end-to-end check ===== > %LOG%
echo started %DATE% %TIME% >> %LOG%
where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)
echo python launcher: %PY% >> %LOG%
%PY% --version >> %LOG% 2>&1
%PY% -c "import numpy; print('numpy', numpy.__version__)" >> %LOG% 2>&1
if errorlevel 1 (
  echo numpy missing, installing it >> %LOG%
  %PY% -m pip install numpy >> %LOG% 2>&1
)

echo.
echo [1/3] Checking your LLM gateway...
echo. >> %LOG%
echo ----- 1. check_llm (real gateway) ----- >> %LOG%
%PY% -m agentx.check_llm >> %LOG% 2>&1

echo [2/3] Running a full session with your real LLM (about 5 minutes)...
echo. >> %LOG%
echo ----- 2. full session with the real LLM (150 days of data) ----- >> %LOG%
echo started %TIME% >> %LOG%
%PY% -m agentx.run_demo --llm http --days 150 --out e2e_out >> %LOG% 2>&1
echo finished %TIME% >> %LOG%

echo [3/3] Running the unit tests...
echo. >> %LOG%
echo ----- 3. unit tests ----- >> %LOG%
%PY% -m unittest discover -s tests -t . -v >> %LOG% 2>&1

echo. >> %LOG%
echo finished %DATE% %TIME% >> %LOG%
echo ALL DONE >> %LOG%
echo.
echo Finished. Tell Claude "done". You can close this window.
pause
