"""Eğitilmiş modelle tek bir görselin sınıfını tahmin eder.

Kullanım:
    python src/predict.py foto.jpg
    python src/predict.py foto.jpg --model models/rps_model.keras
"""

import argparse
import pathlib

import numpy as np
from tensorflow import keras

IMG_SIZE = 150
CLASS_NAMES = ("paper", "rock", "scissors")
TURKCE = {"paper": "Kağıt", "rock": "Taş", "scissors": "Makas"}


def parse_args():
    p = argparse.ArgumentParser(description="Bir görseli taş/kağıt/makas olarak sınıflandırır.")
    p.add_argument("image", help="Tahmin edilecek görselin yolu")
    p.add_argument("--model", default="models/rps_model.keras", help="Model dosyası")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model bulunamadı: {model_path}\n"
            "Önce 'python src/train.py --data-dir <veri_klasoru>' ile modeli eğitin."
        )

    image_path = pathlib.Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Görsel bulunamadı: {image_path}")

    model = keras.models.load_model(model_path)

    img = keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    # Model 0-1 araliginda olceklenmis girdi bekler (egitimde de boyle verildi).
    array = keras.utils.img_to_array(img) / 255.0
    batch = np.expand_dims(array, axis=0)

    probs = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(probs))

    print(f"\nTahmin: {TURKCE[CLASS_NAMES[best]]} ({CLASS_NAMES[best]})  —  güven: %{probs[best] * 100:.1f}\n")
    for name, prob in zip(CLASS_NAMES, probs):
        print(f"  {TURKCE[name]:<6} {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
