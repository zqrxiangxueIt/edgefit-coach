@echo off
rem EdgeFit Coach - full system stop (Pi streamer + dashboard)
cd /d E:\multi_system_design
curl -s -X POST http://127.0.0.1:8000/api/power/off >nul 2>&1
timeout /t 1 /nobreak >nul
rem direct fallback: make sure the Pi streamer is gone even if dashboard was off
ssh -o BatchMode=yes -o ConnectTimeout=8 -i "C:\Users\ZHENG QIAORI\.ssh\id_rsa" saltz@100.93.84.79 "pkill -f 'edge[.]edge_main'" >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
echo System stopped: Pi streamer killed, dashboard closed, camera released.
