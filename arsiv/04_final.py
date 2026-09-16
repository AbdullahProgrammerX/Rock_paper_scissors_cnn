import cv2
import numpy as np
import tensorflow as tf

# --- Ayarlar ---
# En son, veri artırma ile eğittiğimiz dayanıklı modeli kullanıyoruz.
MODEL_PATH = 'rps_model_robust.keras'
CLASS_NAMES = ['paper', 'rock', 'scissors']
IMG_SIZE = 150

# --- Model ve Kamera Kurulumu ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    print("Model ve kamera başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Model veya kamera yüklenemedi. Detay: {e}")
    exit()

# Her ROI için ayrı arka plan çıkarıcılar
bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)


def get_winner(move1, move2):
    """Oyunun kazananını belirler."""
    if move1 == "Bos" or move2 == "Bos":
        return "Oyuncular bekleniyor..."
    if move1 == move2:
        return "Berabere"
    elif (move1 == "rock" and move2 == "scissors") or \
            (move1 == "scissors" and move2 == "paper") or \
            (move1 == "paper" and move2 == "rock"):
        return "Oyuncu 1 Kazandi"
    else:
        return "Oyuncu 2 Kazandi"


def find_and_preprocess_hand(roi, bg_subtractor):
    """
    Bu fonksiyon projenin kalbidir.
    1. Arka planı temizler.
    2. En büyük nesneyi (eli) bulur.
    3. Elin etrafını tam oturacak şekilde kırpar.
    4. Modeli beslemek için hazırlar.
    """
    # 1. Arka planı temizleyerek maske oluştur
    mask = bg_subtractor.apply(roi)

    # Gürültüyü azaltmak için morfolojik işlemler
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    # 2. Maskedeki en büyük konturu (eli) bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        # El olarak kabul etmek için minimum bir alan belirleyelim
        if cv2.contourArea(largest_contour) > 2000:
            # 3. Elin etrafına tam oturan bir kutu çiz (bounding box)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Kırpılmış görüntüyü orijinal renkli ROI'den al
            hand_crop = roi[y:y + h, x:x + w]

            # 4. Modeli beslemek için hazırla (yeniden boyutlandır ve normalize et)
            img_resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            img_batch = np.expand_dims(img_resized, axis=0) / 255.0

            return img_batch, (x, y, w, h)  # Hem işlenmiş resmi hem de kutu koordinatlarını döndür

    return None, None  # El bulunamazsa None döndür


# --- Ana Oyun Döngüsü ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Oyuncu bölgelerini tanımla
    roi1_coords = (50, 100, 300, 400)  # (sol_ust_x, sol_ust_y, sag_alt_x, sag_alt_y)
    roi2_coords = (340, 100, 590, 400)

    # Bölgeleri ana görüntüden ayır
    roi1 = frame[roi1_coords[1]:roi1_coords[3], roi1_coords[0]:roi1_coords[2]]
    roi2 = frame[roi2_coords[1]:roi2_coords[3], roi2_coords[0]:roi2_coords[2]]

    # Oyuncu 1 için eli bul ve tahmin yap
    processed_hand1, bbox1 = find_and_preprocess_hand(roi1, bg_subtractor1)
    move1 = "Bos"
    if processed_hand1 is not None:
        prediction1 = model.predict(processed_hand1, verbose=0)
        move1 = CLASS_NAMES[np.argmax(prediction1)]
        # Tespit edilen elin etrafına DİNAMİK kutuyu çiz (görsel geri bildirim için)
        x, y, w, h = bbox1
        cv2.rectangle(frame, (roi1_coords[0] + x, roi1_coords[1] + y),
                      (roi1_coords[0] + x + w, roi1_coords[1] + y + h), (0, 255, 255), 2)

    # Oyuncu 2 için eli bul ve tahmin yap
    processed_hand2, bbox2 = find_and_preprocess_hand(roi2, bg_subtractor2)
    move2 = "Bos"
    if processed_hand2 is not None:
        prediction2 = model.predict(processed_hand2, verbose=0)
        move2 = CLASS_NAMES[np.argmax(prediction2)]
        x, y, w, h = bbox2
        cv2.rectangle(frame, (roi2_coords[0] + x, roi2_coords[1] + y),
                      (roi2_coords[0] + x + w, roi2_coords[1] + y + h), (255, 0, 255), 2)

    # Kazananı bul ve sonuçları ekrana yazdır
    winner = get_winner(move1, move2)

    cv2.rectangle(frame, (roi1_coords[0], roi1_coords[1]), (roi1_coords[2], roi1_coords[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi2_coords[0], roi2_coords[1]), (roi2_coords[2], roi2_coords[3]), (0, 0, 255), 2)
    cv2.putText(frame, f"Oyuncu 1: {move1}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Oyuncu 2: {move2}", (340, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, winner, (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Akilli Tas-Kagit-Makas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()