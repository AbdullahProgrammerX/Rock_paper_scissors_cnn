import cv2
import numpy as np
import tensorflow as tf

# Modelimizi ve temel ayarları yüklüyoruz (öncekiyle aynı)
model = tf.keras.models.load_model('rps_model_robust.keras')
CLASS_NAMES = ['paper', 'rock', 'scissors']
IMG_SIZE = 150


def get_winner(move1, move2):
    if move1 == move2:
        return "Berabere"
    elif (move1 == "rock" and move2 == "scissors") or \
            (move1 == "scissors" and move2 == "paper") or \
            (move1 == "paper" and move2 == "rock"):
        return "Oyuncu 1 Kazandi"
    else:
        return "Oyuncu 2 Kazandi"


# Kamerayı başlat
cap = cv2.VideoCapture(0)

# YENİ ADIM: Arka Plan Çıkarma nesnelerini oluşturuyoruz.
# Her bir ROI (el bölgesi) için ayrı bir nesne oluşturmak daha stabil sonuçlar verir.
bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI koordinatları (öncekiyle aynı)
    roi1_coords = (50, 100, 300, 400)
    roi2_coords = (340, 100, 590, 400)

    # ROI'leri ana görüntüden kesip alalım
    roi1 = frame[roi1_coords[1]:roi1_coords[3], roi1_coords[0]:roi1_coords[2]]
    roi2 = frame[roi2_coords[1]:roi2_coords[3], roi2_coords[0]:roi2_coords[2]]

    # YENİ ADIM: Her bir ROI'deki arka planı temizleyerek bir 'maske' oluşturalım.
    # Bu maskede sadece hareket eden nesneler (eller) beyaz, geri kalan her şey siyah olacak.
    mask1 = bg_subtractor1.apply(roi1)
    mask2 = bg_subtractor2.apply(roi2)

    # YENİ ADIM: Modelimiz 3 kanallı (RGB) resimlerle eğitildiği için,
    # bu tek kanallı siyah-beyaz maskeyi 3 kanallı hale getiriyoruz.
    mask_3d_1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
    mask_3d_2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)

    # Tahmin için hazırlama (öncekiyle aynı, ama bu sefer maskelenmiş görüntüleri kullanıyoruz)
    img_processed1 = cv2.resize(mask_3d_1, (IMG_SIZE, IMG_SIZE))
    img_batch1 = np.expand_dims(img_processed1, axis=0) / 255.0

    img_processed2 = cv2.resize(mask_3d_2, (IMG_SIZE, IMG_SIZE))
    img_batch2 = np.expand_dims(img_processed2, axis=0) / 255.0

    # Tahminleri yap
    prediction1 = model.predict(img_batch1, verbose=0)
    move1 = CLASS_NAMES[np.argmax(prediction1)]

    prediction2 = model.predict(img_batch2, verbose=0)
    move2 = CLASS_NAMES[np.argmax(prediction2)]

    # Kazananı belirle
    winner = get_winner(move1, move2)

    # Sonuçları ekrana yazdır (öncekiyle aynı)
    cv2.rectangle(frame, (roi1_coords[0], roi1_coords[1]), (roi1_coords[2], roi1_coords[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi2_coords[0], roi2_coords[1]), (roi2_coords[2], roi2_coords[3]), (0, 0, 255), 2)
    cv2.putText(frame, f"Oyuncu 1: {move1}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Oyuncu 2: {move2}", (340, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, winner, (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # İPUCU: Maskelerin nasıl göründüğünü anlamak için bu satırları aktif edebilirsiniz
    # cv2.imshow('Mask 1', mask_3d_1)
    # cv2.imshow('Mask 2', mask_3d_2)

    cv2.imshow('Tas - Kagit - Makas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()