import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# --- Ayarlar ---
MODEL_PATH = 'rps_model_robust.keras'  # Veri artırma ile eğittiğimiz en güçlü model
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


def process_hand_greenscreen(roi, hands_detector):
    """
    BU FONKSİYON PROJENİN ZİRVE NOKTASIDIR:
    1. MediaPipe ile eli net bir şekilde bulur.
    2. Elin bir maskesini oluşturur.
    3. Elin arkasına sanal bir yeşil perde ekler.
    4. En boy oranını koruyarak modeli beslemeye hazırlar.
    """
    frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = roi.shape

        # YENİ ADIM 1: Elin maskesini oluşturma
        # Boş bir siyah tuval oluştur
        mask = np.zeros((h, w), dtype="uint8")

        # Elin dış hatlarını (convex hull) çizerek bir alan oluştur
        landmark_points = np.array([[int(l.x * w), int(l.y * h)] for l in hand_landmarks.landmark])
        convex_hull = cv2.convexHull(landmark_points)
        cv2.fillConvexPoly(mask, convex_hull, 255)  # Alanın içini beyaza boya

        # YENİ ADIM 2: Sanal yeşil perde oluşturma
        green_screen = np.zeros_like(roi)
        green_screen[:] = (0, 255, 0)  # BGR formatında yeşil renk

        # YENİ ADIM 3: Maskeyi kullanarak eli ve yeşil perdeyi birleştirme
        # Maskenin olduğu yerden orijinal el görüntüsünü al
        hand_part = cv2.bitwise_and(roi, roi, mask=mask)
        # Maskenin olmadığı yerden (arka plan) yeşil perdeyi al
        background_part = cv2.bitwise_and(green_screen, green_screen, mask=cv2.bitwise_not(mask))
        # Bu iki parçayı birleştirerek son görüntüyü oluştur
        combined_image = cv2.add(hand_part, background_part)

        # YENİ ADIM 4: En boy oranını koruyarak modeli beslemeye hazırlama
        # (Önceki adımdaki en iyi tekniği burada da kullanıyoruz)
        crop_h, crop_w, _ = combined_image.shape
        max_len = max(crop_w, crop_h)
        square_canvas = np.zeros((max_len, max_len, 3), np.uint8)
        square_canvas[:] = (0, 255, 0)  # Kenar boşluklarını da yeşil yap
        y_offset, x_offset = (max_len - crop_h) // 2, (max_len - crop_w) // 2
        square_canvas[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w] = combined_image

        img_resized = cv2.resize(square_canvas, (IMG_SIZE, IMG_SIZE))
        img_batch = np.expand_dims(img_resized, axis=0) / 255.0

        return img_batch, hand_landmarks, img_resized  # İşlenmiş son halini de geri döndürelim

    return None, None, None


# --- Ana Oyun Döngüsü ---
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    roi1_coords = (50, 100, 300, 400);
    roi2_coords = (340, 100, 590, 400)
    roi1 = frame[roi1_coords[1]:roi1_coords[3], roi1_coords[0]:roi1_coords[2]]
    roi2 = frame[roi2_coords[1]:roi2_coords[3], roi2_coords[0]:roi2_coords[2]]

    # Oyuncu 1
    processed_hand1, landmarks1, final_img1 = process_hand_greenscreen(roi1, hands1)
    move1 = "Tespit Ediliyor..."
    if processed_hand1 is not None:
        prediction1 = model.predict(processed_hand1, verbose=0)
        move1 = CLASS_NAMES[np.argmax(prediction1)]
        mp_drawing.draw_landmarks(roi1, landmarks1, mp_hands.HAND_CONNECTIONS)
        # GÖRSEL DOĞRULAMA: Modelin ne gördüğünü canlı izleyelim
        cv2.imshow('Modelin Gozu 1', final_img1)

    # Oyuncu 2
    processed_hand2, landmarks2, final_img2 = process_hand_greenscreen(roi2, hands2)
    move2 = "Tespit Ediliyor..."
    if processed_hand2 is not None:
        prediction2 = model.predict(processed_hand2, verbose=0)
        move2 = CLASS_NAMES[np.argmax(prediction2)]
        mp_drawing.draw_landmarks(roi2, landmarks2, mp_hands.HAND_CONNECTIONS)
        # GÖRSEL DOĞRULAMA: Modelin ne gördüğünü canlı izleyelim
        cv2.imshow('Modelin Gozu 2', final_img2)

    winner = get_winner(move1, move2)

    # Arayüzü çiz
    cv2.rectangle(frame, (roi1_coords[0], roi1_coords[1]), (roi1_coords[2], roi1_coords[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi2_coords[0], roi2_coords[1]), (roi2_coords[2], roi2_coords[3]), (0, 0, 255), 2)
    cv2.putText(frame, f"Oyuncu 1: {move1}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Oyuncu 2: {move2}", (340, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, winner, (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Profesyonel Tas-Kagit-Makas GREENSCREEN', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()