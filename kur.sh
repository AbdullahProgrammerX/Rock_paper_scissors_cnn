#!/usr/bin/env bash
# Linux / macOS icin kurulum. Sanal ortam olusturur ve bagimliliklari kurar.
set -e

echo "================================================"
echo "  Taş-Kağıt-Makas - Kurulum"
echo "================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[HATA] python3 bulunamadı. Python 3.10-3.12 kurun."
    exit 1
fi

echo "[1/3] Sanal ortam oluşturuluyor..."
[ -d .venv ] || python3 -m venv .venv

echo "[2/3] pip güncelleniyor..."
./.venv/bin/python -m pip install --upgrade pip --quiet

echo "[3/3] Gerekli paketler kuruluyor (birkaç dakika sürebilir)..."
./.venv/bin/python -m pip install -r requirements.txt

echo
echo "================================================"
echo "  Kurulum tamamlandı!"
echo
echo "  Oyunu başlatmak için: ./oyna.sh"
echo "================================================"
