#!/usr/bin/env bash
# Oyunu baslatir. Once ./kur.sh calistirilmis olmalidir.
set -e

if [ ! -x .venv/bin/python ]; then
    echo "[HATA] Önce ./kur.sh dosyasını çalıştırın."
    exit 1
fi

echo "Oyun başlatılıyor... Çıkmak için oyun penceresinde 'q' tuşuna basın."
exec ./.venv/bin/python src/oyun.py "$@"
