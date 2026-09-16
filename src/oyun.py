"""İki kişilik canlı taş-kağıt-makas oyunu.

Kameradaki iki bölgeye (sol/sağ) tutulan eller MediaPipe ile bulunur, eğitilmiş
CNN ile sınıflandırılır ve kazanan ekranda gösterilir. Çıkmak için 'q'.

Kullanım:
    python src/oyun.py
    python src/oyun.py --camera 1 --model models/rps_model.keras
"""

import argparse
import pathlib
import sys

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["paper", "rock", "scissors"]
TURKCE = {"paper": "Kagit", "rock": "Tas", "scissors": "Makas"}
IMG_SIZE = 150
BEKLEME = "Tespit Ediliyor..."

# Kameradaki iki oyuncu bölgesi: (x1, y1, x2, y2)
ROI1 = (50, 100, 300, 400)
ROI2 = (340, 100, 590, 400)


def parse_args():
    p = argparse.ArgumentParser(description="İki kişilik canlı taş-kağıt-makas oyunu.")
    p.add_argument("--model", default="models/rps_model.keras", help="Model dosyası")
    p.add_argument("--camera", type=int, default=0, help="Kamera indeksi (varsayılan 0)")
    return p.parse_args()


def get_winner(move1, move2):
    """Oyunun kazananını belirler."""
    if move1 == BEKLEME or move2 == BEKLEME:
        return "Oyuncular bekleniyor..."
    if move1 == move2:
        return "Berabere"
    if (move1, move2) in (("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")):
        return "Oyuncu 1 Kazandi"
    return "Oyuncu 2 Kazandi"


def process_hand(roi, hands_detector):
    """Bölgedeki eli bulur, en boy oranını bozmadan kırpar ve modele hazırlar.

    El, kare bir tuvalin ortasına yerleştirilir; doğrudan yeniden boyutlandırmak
    eli yatay/dikey eziyor ve tahmini belirgin şekilde bozuyordu.
    """
    results = hands_detector.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    h, w = roi.shape[:2]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    padding = 20
    x_min = max(0, int(min(xs) * w) - padding)
    y_min = max(0, int(min(ys) * h) - padding)
    x_max = min(w, int(max(xs) * w) + padding)
    y_max = min(h, int(max(ys) * h) + padding)

    hand_crop = roi[y_min:y_max, x_min:x_max]
    if hand_crop.size == 0:
        return None, None

    crop_h, crop_w = hand_crop.shape[:2]
    max_len = max(crop_w, crop_h)
    square = np.zeros((max_len, max_len, 3), np.uint8)
    y_off, x_off = (max_len - crop_h) // 2, (max_len - crop_w) // 2
    square[y_off:y_off + crop_h, x_off:x_off + crop_w] = hand_crop

    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE))
    batch = np.expand_dims(resized, axis=0) / 255.0   # Model 0-1 aralığı bekler.
    return batch, hand_landmarks


def main():
    args = parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model bulunamadı: {model_path}\n"
                 "Depoyu eksiksiz klonladığınızdan emin olun (models/rps_model.keras).")

    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Kamera açılamadı (indeks {args.camera}). --camera ile başka bir indeks deneyin.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    # Her oyuncu bölgesi için ayrı dedektör; tek dedektör iki bölge arasında kararsız kalıyor.
    hands1 = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    hands2 = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    print("Oyun başladı. Ellerinizi kutuların içine tutun. Çıkmak için 'q'.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Ayna görüntüsü

        moves = []
        for (x1, y1, x2, y2), detector in ((ROI1, hands1), (ROI2, hands2)):
            roi = frame[y1:y2, x1:x2]
            batch, landmarks = process_hand(roi, detector)
            if batch is None:
                moves.append(BEKLEME)
                continue
            prediction = model.predict(batch, verbose=0)
            moves.append(CLASS_NAMES[int(np.argmax(prediction))])
            mp_drawing.draw_landmarks(roi, landmarks, mp_hands.HAND_CONNECTIONS)

        move1, move2 = moves
        winner = get_winner(move1, move2)

        cv2.rectangle(frame, ROI1[:2], ROI1[2:], (0, 255, 0), 2)
        cv2.rectangle(frame, ROI2[:2], ROI2[2:], (0, 0, 255), 2)
        cv2.putText(frame, f"Oyuncu 1: {TURKCE.get(move1, move1)}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Oyuncu 2: {TURKCE.get(move2, move2)}", (340, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, winner, (220, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Tas-Kagit-Makas", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
