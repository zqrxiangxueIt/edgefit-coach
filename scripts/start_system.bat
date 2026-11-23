@echo off
rem EdgeFit Coach - ONE-CLICK full start: dashboard + auto power-on + browser
cd /d E:\multi_system_design
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
start "EdgeFit Dashboard" /min .venv\Scripts\python.exe server\app.py
echo Starting dashboard...
timeout /t 4 /nobreak >nul
echo Powering on Pi camera + inference...
curl -s -X POST http://127.0.0.1:8000/api/power/on >nul 2>&1
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8000/
echo System ready. To fully stop later, run stop_system.bat (or ask the agent).
