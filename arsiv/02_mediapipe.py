import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# --- Ayarlar ---
MODEL_PATH = 'rps_model_robust.keras'
CLASS_NAMES = ['paper', 'rock', 'scissors']
IMG_SIZE = 150

# --- Model ve MediaPipe Kurulumu ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands1 = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    hands2 = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    print("Model ve MediaPipe başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Kurulum başarısız. Detay: {e}")
    exit()


def get_winner(move1, move2):
    """Oyunun kazananını belirler."""
    if move1 == "Tespit Ediliyor..." or move2 == "Tespit Ediliyor...":
        return "Oyuncular bekleniyor..."
    if move1 == move2:
        return "Berabere"
    elif (move1 == "rock" and move2 == "scissors") or \
            (move1 == "scissors" and move2 == "paper") or \
            (move1 == "paper" and move2 == "rock"):
        return "Oyuncu 1 Kazandi"
    else:
        return "Oyuncu 2 Kazandi"


def process_hand_aspect_ratio(roi, hands_detector):
    """
    BU FONKSİYON GÜNCELLENDİ:
    En boy oranını koruyarak eli bulur, kırpar ve modele hazır hale getirir.
    """
    frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = roi.shape
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

        padding = 20
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

        hand_crop = roi[y_min:y_max, x_min:x_max]

        if hand_crop.size > 0:
            # --- EN BOY ORANINI KORUMA ADIMI ---
            crop_h, crop_w, _ = hand_crop.shape

            # 1. Boş bir kare tuval oluştur (siyah)
            max_len = max(crop_w, crop_h)
            square_canvas = np.zeros((max_len, max_len, 3), np.uint8)

            # 2. Kırpılmış dikdörtgen eli bu kare tuvalin ortasına yerleştir
            y_offset = (max_len - crop_h) // 2
            x_offset = (max_len - crop_w) // 2
            square_canvas[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w] = hand_crop

            # 3. Artık bozulma olmadan yeniden boyutlandırabiliriz
            img_resized = cv2.resize(square_canvas, (IMG_SIZE, IMG_SIZE))
            img_batch = np.expand_dims(img_resized, axis=0) / 255.0

            return img_batch, hand_landmarks

    return None, None


# --- Ana Oyun Döngüsü ---
# (Bu kısım öncekiyle aynı, sadece fonksiyon adı değişti)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    roi1_coords = (50, 100, 300, 400);
    roi2_coords = (340, 100, 590, 400)
    roi1 = frame[roi1_coords[1]:roi1_coords[3], roi1_coords[0]:roi1_coords[2]]
    roi2 = frame[roi2_coords[1]:roi2_coords[3], roi2_coords[0]:roi2_coords[2]]

    # Oyuncu 1
    processed_hand1, landmarks1 = process_hand_aspect_ratio(roi1, hands1)
    move1 = "Tespit Ediliyor..."
    if processed_hand1 is not None:
        prediction1 = model.predict(processed_hand1, verbose=0)
        move1 = CLASS_NAMES[np.argmax(prediction1)]
        mp_drawing.draw_landmarks(roi1, landmarks1, mp_hands.HAND_CONNECTIONS)

    # Oyuncu 2
    processed_hand2, landmarks2 = process_hand_aspect_ratio(roi2, hands2)
    move2 = "Tespit Ediliyor..."
    if processed_hand2 is not None:
        prediction2 = model.predict(processed_hand2, verbose=0)
        move2 = CLASS_NAMES[np.argmax(prediction2)]
        mp_drawing.draw_landmarks(roi2, landmarks2, mp_hands.HAND_CONNECTIONS)

    winner = get_winner(move1, move2)

    # Arayüzü çiz
    cv2.rectangle(frame, (roi1_coords[0], roi1_coords[1]), (roi1_coords[2], roi1_coords[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi2_coords[0], roi2_coords[1]), (roi2_coords[2], roi2_coords[3]), (0, 0, 255), 2)
    cv2.putText(frame, f"Oyuncu 1: {move1}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Oyuncu 2: {move2}", (340, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, winner, (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Profesyonel Tas-Kagit-Makas v2', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()