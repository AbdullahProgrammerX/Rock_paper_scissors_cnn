"""Web kamerasından canlı taş-kağıt-makas tahmini yapar.

Elinizi kameradaki kareye tutun; tahmin ve güven skoru ekranda gösterilir.
Çıkmak için 'q' tuşuna basın.

Kullanım:
    python src/webcam.py
    python src/webcam.py --model models/rps_model.keras --camera 1
"""

import argparse
import pathlib

import cv2
import numpy as np
from tensorflow import keras

IMG_SIZE = 150
CLASS_NAMES = ("paper", "rock", "scissors")
TURKCE = {"paper": "Kagit", "rock": "Tas", "scissors": "Makas"}
BOX = 300  # Ekrandaki tahmin karesinin kenar uzunluğu (piksel)


def parse_args():
    p = argparse.ArgumentParser(description="Web kamerasıyla canlı tahmin.")
    p.add_argument("--model", default="models/rps_model.keras", help="Model dosyası")
    p.add_argument("--camera", type=int, default=0, help="Kamera indeksi (varsayılan 0)")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model bulunamadı: {model_path}\n"
            "Önce 'python src/train.py --data-dir <veri_klasoru>' ile modeli eğitin."
        )

    model = keras.models.load_model(model_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Kamera açılamadı (indeks {args.camera}). --camera ile başka bir indeks deneyin.")

    print("Elinizi yeşil karenin içine tutun. Çıkmak için 'q'.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # Ayna görüntüsü, kullanıcı için daha doğal
        h, w = frame.shape[:2]
        x1, y1 = (w - BOX) // 2, (h - BOX) // 2
        roi = frame[y1:y1 + BOX, x1:x1 + BOX]

        # Modelin beklediği boyuta indir; ölçekleme modelin ilk katmanında yapılıyor.
        resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32")
        probs = model.predict(np.expand_dims(rgb, axis=0), verbose=0)[0]
        best = int(np.argmax(probs))
        label = f"{TURKCE[CLASS_NAMES[best]]}  %{probs[best] * 100:.0f}"

        cv2.rectangle(frame, (x1, y1), (x1 + BOX, y1 + BOX), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Tas-Kagit-Makas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
