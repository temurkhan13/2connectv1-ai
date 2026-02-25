@echo off
REM ============================================================
REM Celery Worker Startup Script for Windows
REM ============================================================
REM
REM IMPORTANT: Windows requires --pool=solo because prefork
REM uses fork() which doesn't work on Windows.
REM
REM Usage:
REM   scripts\start_celery_worker.bat
REM
REM ============================================================

cd /d %~dp0..

echo ============================================================
echo Starting Celery Worker (Windows mode: --pool=solo)
echo ============================================================

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start Celery with solo pool (required for Windows)
celery -A app.core.celery worker --pool=solo -l info

pause
