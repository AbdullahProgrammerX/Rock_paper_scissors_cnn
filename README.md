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

**Depoyla birlikte gelen `models/rps_model.keras` dosyası, bu 4. aşama modelidir** (24 MB).
Önceki aşamaların modelleri, `Flatten → Dense(512)` katmanı nedeniyle ~218 MB olduklarından
depoya dahil edilmemiştir; `src/train.py --model cnn` ile yeniden üretilebilirler.

## Model Mimarisi

**Sıfırdan CNN:** `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dropout(0.5) → Dense(512) → Dense(3, softmax)`

**Transfer learning:** `MobileNetV2 (ImageNet) → GlobalAveragePooling → Dropout(0.5) → Dense(128) → Dense(3, softmax)`

Girdi boyutu 150×150×3, kayıp fonksiyonu `categorical_crossentropy`, optimizer `Adam`.
Takip edilen metrikler: accuracy, precision, recall ve macro F1-score.

---

## Hızlı Başlangıç — Modeli Kullanma

Model **zaten eğitilmiştir**; veri setini indirmenize veya yeniden eğitim yapmanıza gerek yoktur.
Aşağıdaki üç adımla kendi bilgisayarınızda çalıştırabilirsiniz.

### 1. Depoyu klonlayın

```bash
git clone https://github.com/AbdullahProgrammerX/Rock_paper_scissors_cnn.git
cd Rock_paper_scissors_cnn
```

### 2. Bağımlılıkları kurun

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

> Python 3.10–3.12 gereklidir. GPU şart değildir; tahmin için CPU fazlasıyla yeterlidir.

### 3. Modeli çalıştırın

**Eğitilmiş model dosyası: `models/rps_model.keras`** — depoyla birlikte gelir, ayrıca indirmeniz gerekmez.

Tek bir fotoğrafı sınıflandırmak için:

```bash
python src/predict.py foto.jpg
```

```
Tahmin: Makas (scissors)  —  güven: %98.7

  Kağıt    0.61%
  Taş      0.69%
  Makas   98.70%
```

Web kameranızla canlı denemek için:

```bash
python src/webcam.py
```

Elinizi ekrandaki yeşil karenin içine tutun; tahmin ve güven skoru canlı olarak gösterilir.
Çıkmak için `q` tuşuna basın.

### Kendi kodunuzda kullanma

Model standart bir Keras modelidir; doğrudan yükleyip kullanabilirsiniz:

```python
import numpy as np
from tensorflow import keras

CLASS_NAMES = ("paper", "rock", "scissors")   # Sıra önemlidir, modelin çıkış sırasıdır.

model = keras.models.load_model("models/rps_model.keras")

img = keras.utils.load_img("foto.jpg", target_size=(150, 150))
array = keras.utils.img_to_array(img) / 255.0                   # 0-1 aralığına ölçekle
batch = np.expand_dims(array, axis=0)                          # (1, 150, 150, 3)

probs = model.predict(batch)[0]
print(CLASS_NAMES[int(np.argmax(probs))], probs.max())
```

**Girdi:** 150×150×3 RGB görsel, piksel değerleri **0–1 aralığına ölçeklenmiş** (`/255`).
Model bu ölçeklemeyi kendi içinde yapmaz — yukarıdaki gibi dışarıda uygulamanız gerekir.
**Çıktı:** 3 elemanlı softmax olasılık vektörü — sırasıyla `paper`, `rock`, `scissors`.

### İyi sonuç almak için ipuçları

- Eli sade ve tek renk bir zemin önünde tutun (model bu tür verilerle eğitildi).
- El, karenin büyük kısmını doldursun; çok uzaktan çekilen fotoğraflarda başarım düşer.
- Aydınlatma yeterli olsun; aşırı karanlık veya ters ışık tahmini bozar.

---

## Yeniden Eğitim (isteğe bağlı)

Modeli kendi verinizle sıfırdan eğitmek isterseniz bu bölümü izleyin. **Modeli sadece kullanmak
istiyorsanız bu adımlara gerek yoktur.**

### Veri setini hazırlama

Veri setleri boyutları nedeniyle depoya dahil edilmemiştir. Eğitim kodu, sınıf adlarının klasör
adlarından okunduğu standart bir yapı bekler:

```
veri/
├── paper/
├── rock/
└── scissors/
```

**Kaggle veri seti (2188 görsel):**

```bash
pip install kaggle
kaggle datasets download -d drgfreeman/rockpaperscissors -p veri --unzip
```

Veri setinin sayfası: [Rock-Paper-Scissors Images](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)

**Kendi fotoğraflarınız:** Yukarıdaki üç klasörü oluşturup kendi el fotoğraflarınızı ilgili
klasörlere koymanız yeterlidir. Sınıf başına en az ~100 görsel önerilir. Bu projenin nihai modeli,
Kaggle veri seti ile 450 adet gerçek ortam fotoğrafının birleştirilmesiyle (hibrit veri seti)
eğitilmiştir; gerçek dünya performansını belirgin şekilde artıran adım budur.

### Eğitimi başlatma

```bash
# Transfer learning (nihai modelin eğitildiği yöntem)
python src/train.py --data-dir veri

# Sıfırdan CNN (projenin ilk fazı)
python src/train.py --data-dir veri --model cnn --epochs 20
```

Eğitilen model varsayılan olarak `models/rps_model.keras` dosyasına yazılır — mevcut modelin
üzerine yazmamak için `--output models/kendi_modelim.keras` kullanabilirsiniz.

| Argüman | Varsayılan | Açıklama |
|---------|-----------|----------|
| `--data-dir` | *(zorunlu)* | `paper/ rock/ scissors/` alt klasörlerini içeren veri klasörü |
| `--model` | `mobilenet` | `mobilenet` veya `cnn` |
| `--epochs` | `10` | 1. aşama epoch sayısı |
| `--finetune-epochs` | `20` | İnce ayar epoch sayısı |
| `--batch-size` | `32` | Batch boyutu |
| `--output` | `models/rps_model.keras` | Modelin kaydedileceği yol |

Eğitim sırasında en iyi doğrulama başarımına sahip ağırlıklar otomatik kaydedilir
(`ModelCheckpoint`), iyileşme durursa öğrenme oranı düşürülür (`ReduceLROnPlateau`) ve
eğitim erken durdurulur (`EarlyStopping`).

## Proje Yapısı

```
├── models/
│   └── rps_model.keras              # Eğitilmiş model — kullanıma hazır
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
