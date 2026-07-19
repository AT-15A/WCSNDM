@echo off
set PYTHONIOENCODING=utf-8
cd /d %~dp0
python hf_client.py %1 %2
exit /b %errorlevel%
