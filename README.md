# Taş-Kağıt-Makas Görüntü Sınıflandırma (Rock Paper Scissors CNN)

El hareketi fotoğraflarından **taş**, **kağıt** ve **makas** işaretlerini tanıyan bir derin öğrenme projesi.
Proje, sıfırdan yazılan basit bir CNN'den başlayıp veri artırma ve MobileNetV2 tabanlı transfer learning ile
**%99.4 doğrulama başarımına** ulaşan kademeli bir geliştirme sürecini belgeler.

## Sonuçlar

| Aşama | Yaklaşım | Doğrulama Başarımı |
|-------|----------|--------------------|
| 1 | Sıfırdan CNN (3 evrişim bloğu) | %97.7 |
| 2 | + EarlyStopping ile optimizasyon | %98.4 |
| 3 | + Veri artırma (data augmentation) | %99.3 |
| 4 | MobileNetV2 transfer learning + hibrit veri seti | **%99.4** |

Aşama 4'teki model, iki kademeli eğitilir: önce yalnızca yeni sınıflandırıcı katmanları (`lr=1e-3`),
ardından MobileNetV2'nin son 40 katmanı çok düşük öğrenme oranıyla (`lr=1e-5`) ince ayarlanır.

## Model Mimarisi

**Sıfırdan CNN:** `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dropout(0.5) → Dense(512) → Dense(3, softmax)`

**Transfer learning:** `MobileNetV2 (ImageNet) → GlobalAveragePooling → Dropout(0.5) → Dense(128) → Dense(3, softmax)`

Girdi boyutu 150×150×3, kayıp fonksiyonu `categorical_crossentropy`, optimizer `Adam`.
Takip edilen metrikler: accuracy, precision, recall ve macro F1-score.

---

## Kurulum

### 1. Depoyu klonlayın

```bash
git clone https://github.com/AbdullahDOGAN1/rock-paper-scissors-cnn.git
cd rock-paper-scissors-cnn
```

### 2. Sanal ortam oluşturun ve bağımlılıkları kurun

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Python 3.10–3.12 gereklidir. GPU şart değildir; CPU ile de eğitilir, yalnızca daha yavaştır.

### 3. Veri setini hazırlayın

Veri setleri boyutları nedeniyle depoya dahil edilmemiştir. İki seçeneğiniz var:

**A) Kaggle veri seti (2188 görsel):**

[Rock-Paper-Scissors Images](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) veri setini indirip
proje kökünde aşağıdaki yapıyı oluşturacak şekilde açın:

```
veri/
├── paper/
├── rock/
└── scissors/
```

Kaggle CLI ile:
```bash
pip install kaggle
kaggle datasets download -d drgfreeman/rockpaperscissors -p veri --unzip
```

**B) Kendi fotoğraflarınız:**

`veri/paper`, `veri/rock`, `veri/scissors` klasörlerini oluşturup kendi el fotoğraflarınızı
ilgili klasörlere koymanız yeterli. Sınıf başına en az ~100 görsel önerilir.
Bu projenin en iyi modeli, Kaggle veri seti ile 450 adet gerçek ortam fotoğrafının
birleştirilmesiyle (hibrit veri seti) eğitilmiştir — gerçek dünya performansını belirgin şekilde artırır.

---

## Kullanım

### Modeli eğitme

```bash
# Transfer learning (önerilen)
python src/train.py --data-dir veri

# Sıfırdan CNN
python src/train.py --data-dir veri --model cnn --epochs 20
```

Eğitilen model `models/rps_model.keras` dosyasına kaydedilir. Tüm seçenekler için:
`python src/train.py --help`

| Argüman | Varsayılan | Açıklama |
|---------|-----------|----------|
| `--data-dir` | *(zorunlu)* | `paper/ rock/ scissors/` alt klasörlerini içeren veri klasörü |
| `--model` | `mobilenet` | `mobilenet` veya `cnn` |
| `--epochs` | `10` | 1. aşama epoch sayısı |
| `--finetune-epochs` | `20` | İnce ayar epoch sayısı |
| `--batch-size` | `32` | Batch boyutu |
| `--output` | `models/rps_model.keras` | Kayıt yolu |

### Tek bir görseli tahmin etme

```bash
python src/predict.py foto.jpg
```

```
Tahmin: Makas (scissors)  —  güven: %98.7

  Kağıt    0.61%
  Taş      0.69%
  Makas   98.70%
```

### Web kamerasıyla canlı demo

```bash
python src/webcam.py
```

Elinizi ekrandaki yeşil karenin içine tutun; tahmin canlı olarak gösterilir. Çıkmak için `q`.

---

## Proje Yapısı

```
├── src/
│   ├── train.py                     # Yerelde model eğitimi (CNN / MobileNetV2)
│   ├── predict.py                   # Tek görsel tahmini
│   └── webcam.py                    # Canlı kamera demosu
├── Tas_Kagit_Makas_Projesi.ipynb    # Orijinal Colab notebook (tüm deneyler ve çıktılarıyla)
├── tas_kagit_makas_projesi.py       # Notebook'un Colab'e özel dışa aktarımı
├── requirements.txt
└── *.pdf                            # Proje raporları ve sunum dokümanları
```

`src/` altındaki scriptler yerelde çalışacak şekilde yazılmıştır ve Colab/Google Drive gerektirmez.
Notebook ise projenin tüm geliştirme sürecini, eğitim çıktılarını ve grafiklerini kayıt altına alır.

## Dokümanlar

- **Deep_Learning_Rock_Paper_Scissors_Journey.pdf** — projenin uçtan uca anlatımı
- **Model_Evolution_The_RPS_Story.pdf** — model sürümlerinin karşılaştırmalı evrimi
- **işakışı_author_module.pdf** — iş akışı ve modül dokümantasyonu

## Lisans

MIT
