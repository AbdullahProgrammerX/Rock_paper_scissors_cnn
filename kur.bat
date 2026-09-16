@echo off
REM Windows icin tek tiklik kurulum. Sanal ortam olusturur ve bagimliliklari kurar.
setlocal

echo ================================================
echo   Tas-Kagit-Makas - Kurulum
echo ================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi.
    echo python.org/downloads adresinden Python 3.10-3.12 kurun.
    echo Kurulum sirasinda "Add Python to PATH" secenegini isaretlemeyi unutmayin.
    pause
    exit /b 1
)

echo [1/3] Sanal ortam olusturuluyor...
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 goto hata
)

echo [2/3] pip guncelleniyor...
call .venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [3/3] Gerekli paketler kuruluyor (bu birkac dakika surebilir)...
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto hata

echo.
echo ================================================
echo   Kurulum tamamlandi!
echo.
echo   Oyunu baslatmak icin: oyna.bat
echo ================================================
pause
exit /b 0

:hata
echo.
echo [HATA] Kurulum basarisiz oldu.
echo Uzun dosya yolu hatasi aldiysaniz projeyi C:\rps gibi kisa bir klasore tasiyin.
echo Detaylar icin README.md dosyasindaki "Windows'ta kurulum hata verirse" bolumune bakin.
pause
exit /b 1
