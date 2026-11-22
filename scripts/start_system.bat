@echo off
rem EdgeFit Coach - full system start (dashboard + browser; Pi via power button)
cd /d E:\multi_system_design
rem stop any stale dashboard on port 8000
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
start "EdgeFit Dashboard" /min .venv\Scripts\python.exe server\app.py
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000/
echo Dashboard started. Click the POWER button on the page to start Pi streaming.
