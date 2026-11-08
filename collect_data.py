import csv
import cv2
import time

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# --------------
# Mediapipe
# --------------

base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)

# ------------
# Webcam
# ------------

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# -------------
# Fitxer CSV
# -------------
saved_letters = {}
samples_saved = 0

csv_file = open("data/dataset.csv", "a", newline="")
writer = csv.writer(csv_file)

with open("data/dataset.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 0:
            label = row[0]
            saved_letters[label] = saved_letters.get(label, 0) + 1
            samples_saved += 1


current_label = None

# ----------------
# Bucle principal
# ----------------

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = detector.detect(mp_image)

    h, w, _ = frame.shape

    # -------------
    # Detectar mans
    # -------------

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        raw_landmarks = []

        # Dibuixar els punts
        for lm in hand:
            cx = int(lm.x * w)
            cy = int(lm.y * h)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            raw_landmarks.append((lm.x, lm.y, lm.z))
        
        # ----------------------
        # Normalitzar els punts
        # ----------------------

        wrist_x, wrist_y, wrist_z = raw_landmarks[0]

        normalized = []

        for x, y, z in raw_landmarks:
            normalized.extend([
                x - wrist_x,
                y - wrist_y,
                z - wrist_z
            ])
            
    # -----------------
    # Interfaç
    # -----------------

    cv2.putText(
            frame,
            f"{current_label}: {saved_letters[current_label]}" if current_label else "",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50, 20, 0),
            2
            )

    cv2.putText(
            frame,
            f"Total: {samples_saved}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 20, 0),
            2
            )

    cv2.imshow("Colleccio de dades", frame)

    # --------------------
    # Teclat
    # --------------------

    key = cv2.waitKey(1) & 0xFF

    # Sortir
    if key == ord('1'):
        break

    # A - Z
    elif 65 <= key <= 90 or 97 <= key <= 122:
        current_label = chr(key).upper()

        writer.writerow([current_label] + normalized)

        saved_letters[current_label] = saved_letters.get(current_label, 0) + 1
        samples_saved += 1


cap.release()
csv_file.close()
cv2.destroyAllWindows()
