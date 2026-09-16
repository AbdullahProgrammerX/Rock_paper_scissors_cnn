"""Two-player live rock-paper-scissors game.

Hands held inside the two on-screen zones are located with MediaPipe, classified
with the trained CNN, and the winner is shown on screen.

Both zones are processed on every frame; the result updates instantly below them.

Controls:
    Q      quit

Usage:
    python src/oyun.py
    python src/oyun.py --camera 1 --width 1600 --height 900
"""

import argparse
import collections
import pathlib
import sys

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["paper", "rock", "scissors"]
LABELS = {"paper": "PAPER", "rock": "ROCK", "scissors": "SCISSORS"}
IMG_SIZE = 150

FONT = cv2.FONT_HERSHEY_DUPLEX

# BGR renkler
WHITE = (255, 255, 255)
DIM = (170, 170, 170)
DARK = (28, 28, 28)
GREEN = (90, 220, 120)
RED = (90, 90, 235)
AMBER = (60, 190, 250)

SMOOTHING = 7          # Tahmin kaç kare üzerinden oylanacak
MIN_CONFIDENCE = 0.60  # Bunun altındaki tahminler gösterilmez


def parse_args():
    p = argparse.ArgumentParser(description="Two-player live rock-paper-scissors.")
    p.add_argument("--model", default="models/rps_model.keras", help="Model file")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--width", type=int, default=1280, help="Capture width (default 1280)")
    p.add_argument("--height", type=int, default=720, help="Capture height (default 720)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Çizim yardımcıları
# --------------------------------------------------------------------------- #

def panel(frame, x1, y1, x2, y2, alpha=0.55, color=DARK):
    """Yarı saydam arka plan kutusu — yazıların her zeminde okunmasını sağlar."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color, np.uint8), alpha, roi, 1 - alpha, 0, roi)


def text_size(text, scale, thickness):
    return cv2.getTextSize(text, FONT, scale, thickness)[0]


def put_text(frame, text, x, y, scale=0.7, color=WHITE, thickness=1, anchor="left"):
    """Metni yazar. anchor: left | center | right — x buna göre yorumlanır."""
    tw, _ = text_size(text, scale, thickness)
    if anchor == "center":
        x -= tw // 2
    elif anchor == "right":
        x -= tw
    # Metni her zaman kare içinde tut; ekran görüntüsündeki taşma bu yüzden oluyordu.
    x = max(8, min(x, frame.shape[1] - tw - 8))
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return x


def confidence_bar(frame, x, y, width, value, color, height=6):
    """Güven skorunu ince bir çubukla gösterir."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 70, 70), -1)
    filled = int(width * max(0.0, min(1.0, value)))
    if filled:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)


def rounded_zone(frame, box, color, active):
    """Oyuncu bölgesini köşe işaretleriyle çizer — düz dikdörtgenden daha okunur."""
    x1, y1, x2, y2 = box
    length = max(24, (x2 - x1) // 6)
    thickness = 3 if active else 2
    shade = color if active else tuple(int(c * 0.55) for c in color)
    for cx, dx in ((x1, 1), (x2, -1)):
        for cy, dy in ((y1, 1), (y2, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * length, cy), shade, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * length), shade, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Oyun mantığı
# --------------------------------------------------------------------------- #

def winner_of(move1, move2):
    """(sonuç metni, kazanan oyuncu) döndürür. Kazanan yoksa 0."""
    if move1 is None and move2 is None:
        return "WAITING FOR BOTH PLAYERS", 0
    if move1 is None:
        return "WAITING FOR PLAYER 1", 0
    if move2 is None:
        return "WAITING FOR PLAYER 2", 0
    if move1 == move2:
        return "DRAW", 0
    if (move1, move2) in (("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")):
        return "PLAYER 1 WINS", 1
    return "PLAYER 2 WINS", 2


def crop_hand(roi, detector):
    """Bölgedeki eli bulur, en boy oranını bozmadan kırpar ve modele hazırlar.

    El, kare bir tuvalin ortasına yerleştirilir; doğrudan yeniden boyutlandırmak
    eli yatay/dikey eziyor ve tahmini belirgin şekilde bozuyordu.
    """
    results = detector.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None, None

    landmarks = results.multi_hand_landmarks[0]
    h, w = roi.shape[:2]
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    pad = 20
    x_min = max(0, int(min(xs) * w) - pad)
    y_min = max(0, int(min(ys) * h) - pad)
    x_max = min(w, int(max(xs) * w) + pad)
    y_max = min(h, int(max(ys) * h) + pad)

    hand = roi[y_min:y_max, x_min:x_max]
    if hand.size == 0:
        return None, None

    ch, cw = hand.shape[:2]
    side = max(cw, ch)
    square = np.zeros((side, side, 3), np.uint8)
    square[(side - ch) // 2:(side - ch) // 2 + ch,
           (side - cw) // 2:(side - cw) // 2 + cw] = hand

    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE))
    # OpenCV BGR verir, model RGB ile eğitildi. Dönüşüm atlanırsa kırmızı ve mavi
    # kanallar yer değiştirir; ölçümde doğruluk %96.7'den %94.2'ye düşüyor.
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0) / 255.0, landmarks


class Player:
    """Bir oyuncunun bölgesi ve son tahminleri."""

    def __init__(self, name, color, detector):
        self.name = name
        self.color = color
        self.detector = detector
        self.history = collections.deque(maxlen=SMOOTHING)
        self.move = None          # Yumuşatılmış tahmin
        self.confidence = 0.0
        self.box = (0, 0, 0, 0)

    def update(self, frame, model):
        x1, y1, x2, y2 = self.box
        roi = frame[y1:y2, x1:x2]
        batch, landmarks = crop_hand(roi, self.detector)

        if batch is None:
            self.history.append(None)
        else:
            probs = model.predict(batch, verbose=0)[0]
            idx = int(np.argmax(probs))
            self.history.append((CLASS_NAMES[idx], float(probs[idx])))
            mp.solutions.drawing_utils.draw_landmarks(
                roi, landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(self.color, 1, 2),
                mp.solutions.drawing_utils.DrawingSpec((220, 220, 220), 1, 1),
            )

        self._smooth()

    def _smooth(self):
        """Son karelerin çoğunluk oyu — tek karelik yanlış tahminlerin titremesini keser."""
        votes = [h for h in self.history if h is not None]
        if len(votes) < max(2, self.history.maxlen // 2):
            self.move, self.confidence = None, 0.0
            return

        counts = collections.Counter(name for name, _ in votes)
        best, _ = counts.most_common(1)[0]
        scores = [conf for name, conf in votes if name == best]
        confidence = sum(scores) / len(scores)

        if confidence < MIN_CONFIDENCE:
            self.move, self.confidence = None, confidence
        else:
            self.move, self.confidence = best, confidence


# --------------------------------------------------------------------------- #

def draw_hud(frame, players, result_text, winner):
    h, w = frame.shape[:2]
    # Tüm yazılar kamera genişliğine göre ölçeklenir; sabit boyutlar 640x480'de
    # başlık ve kısayolların üst üste binmesine yol açıyordu.
    s = max(0.62, min(1.25, w / 1280.0))

    # --- Üst bar: başlık ve kısayol ---
    bar_h = int(54 * s)
    panel(frame, 0, 0, w, bar_h, alpha=0.65)
    baseline = int(bar_h * 0.66)

    title = "ROCK  PAPER  SCISSORS"
    hint = "Q  quit"
    pad = int(20 * s)

    title_w = text_size(title, 0.8 * s, 1)[0]
    hint_w = text_size(hint, 0.55 * s, 1)[0]

    put_text(frame, title, pad, baseline, 0.8 * s, WHITE, 1)
    if pad + title_w + pad + hint_w < w - pad:
        put_text(frame, hint, w - pad, baseline, 0.55 * s, DIM, 1, anchor="right")

    # --- Oyuncu bölgeleri ve etiketleri ---
    for index, player in enumerate(players, start=1):
        x1, y1, x2, y2 = player.box
        # Bir kazanan varsa yalnızca onun bölgesi tam parlaklıkta çizilir.
        active = winner == index if winner else True
        rounded_zone(frame, player.box, player.color, active)

        # Etiket paneli kutunun hemen ÜSTÜNDE ve kutu genişliğinde — taşma olmaz.
        label_h = int(58 * s)
        label_bottom = y1 - int(10 * s)
        label_top = label_bottom - label_h
        panel(frame, x1, label_top, x2, label_bottom, alpha=0.6)

        inner = int(12 * s)
        put_text(frame, player.name, x1 + inner, label_top + int(22 * s), 0.55 * s, player.color, 1)

        move_text = LABELS[player.move] if player.move else "SHOW YOUR HAND"
        colour = WHITE if player.move else DIM
        put_text(frame, move_text, x1 + inner, label_bottom - int(18 * s), 0.7 * s, colour, 1)

        if player.move:
            put_text(frame, f"{player.confidence * 100:.0f}%", x2 - inner,
                     label_bottom - int(18 * s), 0.55 * s, DIM, 1, anchor="right")

        bar_h = max(3, int(6 * s))
        confidence_bar(frame, x1 + inner, label_bottom - bar_h - int(4 * s),
                       (x2 - x1) - 2 * inner,
                       player.confidence if player.move else 0.0, player.color, bar_h)

    # --- Sonuç: bölgelerin hemen altında, ekranın dibinde değil ---
    zone_bottom = players[0].box[3]
    band_top = zone_bottom + int(14 * s)
    band_h = int(78 * s)
    band_left = players[0].box[0]
    band_right = players[1].box[2]
    # Kamera alçak çözünürlükteyse bant alta taşabilir; o durumda yukarı çekilir.
    if band_top + band_h > h:
        band_top = max(zone_bottom, h - band_h)

    panel(frame, band_left, band_top, band_right, band_top + band_h, alpha=0.7)
    centre = (band_left + band_right) // 2

    colour = GREEN if winner == 1 else RED if winner == 2 else AMBER
    put_text(frame, result_text, centre, band_top + int(42 * s), 1.0 * s, colour, 2,
             anchor="center")

    if all(p.move for p in players):
        moves = f"{LABELS[players[0].move]}  vs  {LABELS[players[1].move]}"
    else:
        moves = "Both players: keep one hand inside your zone"
    put_text(frame, moves, centre, band_top + int(66 * s), 0.55 * s, DIM, 1, anchor="center")


def layout(players, w, h):
    """İki bölgeyi kare oranını bozmadan, karenin ortasında yan yana yerleştirir.

    Bölgeler bilerek ekranın uçlarına değil ortaya yakın konur: uçlarda oyuncular
    ellerini yana doğru uzatmak zorunda kalıyor ve kenarlar (pencere, lamba)
    genelde ters ışık aldığı için el tespiti başarısız oluyordu.
    """
    size = int(min(h * 0.50, w * 0.33))
    gap = int(w * 0.03)
    top = int(h * 0.22)
    left = (w - (2 * size + gap)) // 2
    players[0].box = (left, top, left + size, top + size)
    players[1].box = (left + size + gap, top, left + 2 * size + gap, top + size)


def main():
    args = parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}\n"
                 "Make sure you cloned the repository completely (models/rps_model.keras).")

    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera (index {args.camera}). Try --camera 1.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    hands = mp.solutions.hands
    # Her bölge için ayrı dedektör; tek dedektör iki bölge arasında kararsız kalıyor.
    players = [
        Player("PLAYER 1", GREEN, hands.Hands(max_num_hands=1, min_detection_confidence=0.7)),
        Player("PLAYER 2", RED, hands.Hands(max_num_hands=1, min_detection_confidence=0.7)),
    ]

    window = "Rock Paper Scissors"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, actual_w, actual_h)

    print("Ready. Both players: hold one hand inside your zone. Q = quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Ayna görüntüsü
        h, w = frame.shape[:2]
        layout(players, w, h)

        # İki bölge de her karede işlenir; sonuç beklemeden anında güncellenir.
        for player in players:
            player.update(frame, model)

        result_text, winner = winner_of(players[0].move, players[1].move)
        draw_hud(frame, players, result_text, winner)
        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
