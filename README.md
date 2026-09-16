# Taş-Kağıt-Makas — Yapay Zekâ ile Canlı Oyun

Web kameranıza el işareti yapın, yapay zekâ ne yaptığınızı anlasın ve kazananı söylesin.
İki kişi aynı anda oynayabilir: ekrandaki iki kutuya ellerinizi tutmanız yeterli.

Arkasında, el fotoğraflarından **taş / kağıt / makas** ayırt etmeyi öğrenen ve
**%99.4 doğrulama başarımına** ulaşan bir derin öğrenme modeli çalışır.
Model eğitilmiş olarak depoyla birlikte gelir — ayrıca indirmeniz veya eğitmeniz gerekmez.

---

## Oynamak için 3 adım

### 1. Projeyi indirin

Git kuruluysa:

```bash
git clone https://github.com/AbdullahProgrammerX/Rock_paper_scissors_cnn.git
cd Rock_paper_scissors_cnn
```

Git yoksa: sayfanın üstündeki yeşil **Code → Download ZIP** düğmesiyle indirip klasöre çıkarın.

> **Windows'ta önemli:** Projeyi `C:\rps` gibi **kısa bir yola** çıkarın. İç içe klasörlerde
> tutarsanız kurulum "uzun dosya yolu" hatası verebilir.

### 2. Kurulumu çalıştırın

Bu adım Python ortamını hazırlar ve gerekli her şeyi kurar. Birkaç dakika sürer (~600 MB indirir).

| İşletim sistemi | Yapılacak |
|-----------------|-----------|
| **Windows** | `kur.bat` dosyasına **çift tıklayın** |
| **Linux / macOS** | Terminalde `./kur.sh` |

Öncesinde bilgisayarınızda **Python 3.10 – 3.12** kurulu olmalıdır
([python.org/downloads](https://www.python.org/downloads/) — Windows'ta kurulum sırasında
*"Add Python to PATH"* kutusunu işaretleyin). Python 3.13 henüz desteklenmiyor.

### 3. Oynayın

| İşletim sistemi | Yapılacak |
|-----------------|-----------|
| **Windows** | `oyna.bat` dosyasına **çift tıklayın** |
| **Linux / macOS** | Terminalde `./oyna.sh` |

Kamera HD çözünürlükte açılır. **Player 1** soldaki yeşil bölgeye, **Player 2** sağdaki kırmızı
bölgeye elini tutar. Hamleler, güven skorlarıyla birlikte canlı olarak üstte görünür.

| Tuş | İşlev |
|-----|-------|
| `SPACE` | Tur başlatır: 3-2-1 geri sayım, ardından hamleler kilitlenir ve puan verilir |
| `R` | Skoru sıfırlar |
| `Q` | Çıkar |

Skor üst barda ortada durur. Turu kazanan oyuncunun bölgesi sonuç ekranında vurgulanır.

Kamera çözünürlüğünü değiştirebilirsiniz:

```bash
oyna.bat --width 1600 --height 900      # Windows
./oyna.sh --width 1600 --height 900     # Linux / macOS
```

**İyi tanıma için ipuçları**

- Elinizi kutunun içinde, kameraya yakın tutun ve kutuyu doldurun.
- Ortam yeterince aydınlık olsun; arkadan gelen güçlü ışık tanımayı bozar.
- Sade bir arka plan (düz duvar) en iyi sonucu verir.
- Hamlenizi net yapın: taş = kapalı yumruk, kağıt = açık avuç, makas = iki parmak.
- Tahmin son 7 karenin oyuyla belirlenir; bir anlık yanlış okuma ekrana yansımaz.
  Güven %60'ın altındaysa hamle "SHOW YOUR HAND" olarak kalır.

---

## Sorun giderme

<details>
<summary><b>Kamera açılmıyor / siyah ekran</b></summary>

Birden fazla kameranız varsa indeksi değiştirin:

```bash
oyna.bat --camera 1      # Windows
./oyna.sh --camera 1     # Linux / macOS
```

Kameranın Zoom, Teams gibi başka bir uygulama tarafından kullanılmadığından emin olun.
Windows'ta *Ayarlar → Gizlilik ve güvenlik → Kamera* altında masaüstü uygulamalarının
kamera erişimi açık olmalıdır.

</details>

<details>
<summary><b>Kurulum "uzun dosya yolu" hatası veriyor (Windows)</b></summary>

TensorFlow'un içindeki bazı dosya yolları 260 karakteri aşar. Hata şuna benzer:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory:
'...\tensorflow\include\external\boringssl\...'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```

İki çözümden biri:

1. **Projeyi kısa bir yola taşıyın** — örneğin `C:\rps`. En kolay çözüm budur.
2. **Uzun yol desteğini açın** — PowerShell'i *yönetici olarak* açıp şunu çalıştırın,
   ardından bilgisayarı yeniden başlatın:

   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
   ```

</details>

<details>
<summary><b>"module 'mediapipe' has no attribute 'solutions'" hatası</b></summary>

Yanlış mediapipe sürümü kurulmuş demektir: MediaPipe 1.0 ile oyunun kullandığı `solutions`
API'si kaldırıldı. `requirements.txt` doğru sürümleri sabitler, onunla kurun:

```bash
pip install -r requirements.txt
```

Elle kuracaksanız: `pip install "mediapipe==0.10.21" "opencv-python==4.11.*"`

</details>

<details>
<summary><b>"python bulunamadı" (Windows)</b></summary>

Python kurulu değil ya da PATH'e eklenmemiş. [python.org/downloads](https://www.python.org/downloads/)
adresinden 3.12 sürümünü kurun ve kurulum ekranındaki **"Add Python to PATH"** kutusunu işaretleyin.
Kurulumdan sonra komut istemini kapatıp yeniden açın.

</details>

<details>
<summary><b>Oyun yavaş çalışıyor</b></summary>

Tahminler CPU üzerinde yapılır; bu normaldir ve oyun için yeterlidir. GPU gerekmez.
Eski bir bilgisayarda takılma yaşarsanız arka plandaki diğer uygulamaları kapatmak yardımcı olur.

</details>

---

## Projenin geri kalanı

Oyun dışında, modeli kendi işinizde kullanmak veya yeniden eğitmek isterseniz.

### Tek bir fotoğrafı sınıflandırma

```bash
.venv\Scripts\python src\predict.py foto.jpg      # Windows
./.venv/bin/python src/predict.py foto.jpg        # Linux / macOS
```

```
Tahmin: Makas (scissors)  —  güven: %98.7

  Kağıt    0.61%
  Taş      0.69%
  Makas   98.70%
```

### Modeli kendi kodunuzda kullanma

```python
import numpy as np
from tensorflow import keras

CLASS_NAMES = ("paper", "rock", "scissors")   # Sıra önemlidir, modelin çıkış sırasıdır.

model = keras.models.load_model("models/rps_model.keras")

img = keras.utils.load_img("foto.jpg", target_size=(150, 150))
array = keras.utils.img_to_array(img) / 255.0   # 0-1 aralığına ölçekle
batch = np.expand_dims(array, axis=0)           # (1, 150, 150, 3)

probs = model.predict(batch)[0]
print(CLASS_NAMES[int(np.argmax(probs))], probs.max())
```

**Girdi:** 150×150×3 RGB, piksel değerleri **0–1 aralığına ölçeklenmiş** (`/255`).
Model bu ölçeklemeyi içinde yapmaz, dışarıda uygulamanız gerekir.
**Çıktı:** 3 elemanlı softmax olasılık vektörü — sırasıyla `paper`, `rock`, `scissors`.

### Kendi verinizi toplama

Kameradan, elin etrafı kırpılmış eğitim görselleri kaydeder (`s` kaydet, `q` çık):

```bash
python src/veri_topla.py --etiket rock --sayi 150
python src/veri_topla.py --etiket paper
python src/veri_topla.py --etiket scissors
```

### Modeli yeniden eğitme

Veri setleri boyutları nedeniyle depoda değildir. Eğitim kodu şu yapıyı bekler:

```
veri/
├── paper/
├── rock/
└── scissors/
```

Kaggle veri seti (2188 görsel):

```bash
pip install kaggle
kaggle datasets download -d drgfreeman/rockpaperscissors -p veri --unzip
```

Eğitimi başlatın:

```bash
python src/train.py --data-dir veri                 # MobileNetV2 (nihai yöntem)
python src/train.py --data-dir veri --model cnn     # Sıfırdan CNN (ilk faz)
```

| Argüman | Varsayılan | Açıklama |
|---------|-----------|----------|
| `--data-dir` | *(zorunlu)* | `paper/ rock/ scissors/` alt klasörlerini içeren klasör |
| `--model` | `mobilenet` | `mobilenet` veya `cnn` |
| `--epochs` | `10` | 1. aşama epoch sayısı |
| `--finetune-epochs` | `20` | İnce ayar epoch sayısı |
| `--batch-size` | `32` | Batch boyutu |
| `--output` | `models/rps_model.keras` | Kayıt yolu (mevcut modelin üzerine yazmamak için değiştirin) |

---

## Modelin gelişimi

| Aşama | Yaklaşım | Doğrulama Başarımı |
|-------|----------|--------------------|
| 1 | Sıfırdan CNN (3 evrişim bloğu) | %97.7 |
| 2 | + EarlyStopping ile optimizasyon | %98.4 |
| 3 | + Veri artırma (data augmentation) | %99.3 |
| 4 | MobileNetV2 transfer learning + hibrit veri seti | **%99.4** |

Hibrit veri seti = Kaggle'daki 2188 görsel + kamerayla toplanan 450 gerçek ortam fotoğrafı.
Gerçek dünya performansını asıl artıran adım bu olmuştur: yalnızca laboratuvar koşullarında
çekilmiş veriyle eğitilen model, gerçek bir odada zorlanıyordu.

**Mimariler**

- *Sıfırdan CNN:* `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dropout(0.5) → Dense(512) → Dense(3, softmax)`
- *Transfer learning:* `MobileNetV2 (ImageNet) → GlobalAveragePooling → Dropout(0.5) → Dense(128) → Dense(3, softmax)`

4. aşama iki kademede eğitilir: önce yalnızca yeni sınıflandırıcı katmanları (`lr=1e-3`),
ardından MobileNetV2'nin son 40 katmanı çok düşük öğrenme oranıyla (`lr=1e-5`) ince ayarlanır.

Depodaki `models/rps_model.keras` bu 4. aşama modelidir (24 MB). Önceki aşamaların modelleri
`Flatten → Dense(512)` katmanı yüzünden ~218 MB olduğundan depoya dahil edilmemiştir;
`src/train.py --model cnn` ile yeniden üretilebilirler.

---

## Proje yapısı

```
├── kur.bat / kur.sh                 # Kurulum (bir kez çalıştırılır)
├── oyna.bat / oyna.sh               # Oyunu başlatır
├── models/
│   └── rps_model.keras              # Eğitilmiş model — kullanıma hazır
├── src/
│   ├── oyun.py                      # İki kişilik canlı oyun (MediaPipe + CNN, skorlu)
│   ├── predict.py                   # Tek görsel tahmini
│   ├── veri_topla.py                # Kamerayla eğitim verisi toplama
│   └── train.py                     # Model eğitimi
├── arsiv/                           # Oyunun denenmiş önceki sürümleri
├── Tas_Kagit_Makas_Projesi.ipynb    # Orijinal Colab notebook (tüm deneyler ve çıktılar)
├── tas_kagit_makas_projesi.py       # Notebook'un Colab'e özel dışa aktarımı
└── *.pdf                            # Proje raporları
```

`src/` altındaki her şey yerelde çalışır; Colab veya Google Drive gerekmez.
Notebook ise geliştirme sürecinin tamamını, eğitim çıktıları ve grafikleriyle birlikte saklar.

## Dokümanlar

- **Deep_Learning_Rock_Paper_Scissors_Journey.pdf** — projenin uçtan uca anlatımı
- **Model_Evolution_The_RPS_Story.pdf** — model sürümlerinin karşılaştırmalı evrimi
- **işakışı_author_module.pdf** — iş akışı ve modül dokümantasyonu

## Lisans

MIT
