@echo off
REM Oyunu baslatir. Once kur.bat calistirilmis olmalidir.
setlocal

if not exist .venv\Scripts\python.exe (
    echo [HATA] Once kur.bat dosyasini calistirin.
    pause
    exit /b 1
)

echo Oyun baslatiliyor... Kamera aciliyor, lutfen bekleyin.
echo Cikmak icin oyun penceresinde 'q' tusuna basin.
echo.
call .venv\Scripts\python.exe src\oyun.py %*
pause
