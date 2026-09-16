"""Web kamerasıyla kendi eğitim verinizi toplar.

MediaPipe ile elin etrafındaki kutu bulunur; 's' tuşuna bastığınızda yalnızca
kırpılmış el görüntüsü ilgili sınıf klasörüne kaydedilir. Çıkmak için 'q'.

Bu projenin nihai modeli, Kaggle veri setine bu yöntemle toplanan 450 gerçek
ortam fotoğrafının eklenmesiyle eğitilmiştir.

Kullanım:
    python src/veri_topla.py --etiket rock
    python src/veri_topla.py --etiket paper --sayi 200 --cikti veri
"""

import argparse
import pathlib
import sys

import cv2
import mediapipe as mp


def parse_args():
    p = argparse.ArgumentParser(description="Kamerayla eğitim verisi toplar.")
    p.add_argument("--etiket", required=True, choices=["rock", "paper", "scissors"],
                   help="Toplanacak sınıf")
    p.add_argument("--sayi", type=int, default=150, help="Hedef görsel sayısı (varsayılan 150)")
    p.add_argument("--cikti", default="veri", help="Veri klasörü (varsayılan: veri)")
    p.add_argument("--camera", type=int, default=0, help="Kamera indeksi")
    return p.parse_args()


def main():
    args = parse_args()

    hedef_dir = pathlib.Path(args.cikti) / args.etiket
    hedef_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Kamera açılamadı (indeks {args.camera}).")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    # Mevcut görsellerin üzerine yazmamak için sayaç klasördeki dosya sayısından başlar.
    sayac = len(list(hedef_dir.glob("*.jpg")))
    print(f"'{hedef_dir}' klasöründe {sayac} görsel var, {args.sayi} adete tamamlanacak.")
    print("Kaydetmek için 's', çıkmak için 'q'.")

    while sayac < args.sayi:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        kutu = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            h, w = frame.shape[:2]
            padding = 20
            x_min = max(0, int(min(p.x for p in lm) * w) - padding)
            y_min = max(0, int(min(p.y for p in lm) * h) - padding)
            x_max = min(w, int(max(p.x for p in lm) * w) + padding)
            y_max = min(h, int(max(p.y for p in lm) * h) + padding)
            kutu = (x_min, y_min, x_max, y_max)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.putText(frame, f"{args.etiket}: {sayac}/{args.sayi} - kayit icin 's'", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Veri Toplama", frame)

        # waitKey tek yerde okunur; iki ayrı çağrı tuş basışlarının kaçmasına yol açıyordu.
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("q"):
            break
        if tus == ord("s") and kutu is not None:
            x_min, y_min, x_max, y_max = kutu
            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.size > 0:
                yol = hedef_dir / f"{args.etiket}_{sayac}.jpg"
                cv2.imwrite(str(yol), hand_crop)
                print(f"{yol} kaydedildi.")
                sayac += 1

    print(f"\nToplam {sayac} görsel '{hedef_dir}' klasöründe.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
