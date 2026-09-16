# Arşiv — Oyunun Gelişim Aşamaları

Bu klasör, oyunun nihai haline (`src/oyun.py`) ulaşana kadar denenen sürümleri saklar.
Her biri, el bölgesini kameradan ayıklamak için farklı bir yaklaşım dener:

| Dosya | Yaklaşım | Neden yetersiz kaldı |
|-------|----------|----------------------|
| `01_arkaplan_cikarma.py` | `createBackgroundSubtractorMOG2` ile hareketli bölgeyi ayıklama | El sabit durunca arka plana karışıyor, kamera/ışık değişiminde bozuluyor |
| `02_mediapipe.py` | MediaPipe ile el iskeletinden kırpma | Kırpılan dikdörtgen kareye zorlanınca el eziliyor, tahmin bozuluyor |
| `03_greenscreen.py` | El maskesini çıkarıp düz zemine yerleştirme | Maske kenarları gürültülü, model eğitim verisinden uzaklaşıyor |
| `04_final.py` | 02'nin iyileştirilmiş hali | `rps_model_robust.keras` (218 MB) ile çalışır |

**Bu dosyalar olduğu gibi çalışmaz:** hepsi `rps_model_robust.keras` dosyasını arar.
Bu model 218 MB olduğu için depoya dahil edilmemiştir (GitHub'ın dosya başına 100 MB sınırı).
Üretmek isterseniz:

```bash
python src/train.py --data-dir veri --model cnn --output rps_model_robust.keras
```

Çalışan güncel sürüm için `src/oyun.py` dosyasını kullanın — nihai `models/rps_model.keras`
modeliyle birlikte doğrudan çalışır.
