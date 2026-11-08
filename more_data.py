import csv
import cv2
from collections import deque

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
ij_file = open("data/ij.csv", "a", newline="")
uv_file = open("data/uv.csv", "a", newline="")
cc_file = open("data/cc.csv", "a", newline="")
ij_writer = csv.writer(ij_file)
uv_writer = csv.writer(uv_file)
cc_writer = csv.writer(cc_file)


# -------------
# Variables
# -------------
current_label = None
wrist_buffer = deque(maxlen=20)
index_buffer = deque(maxlen=20)
thumb_buffer = deque(maxlen=20)

# -------------
# Funcions
# -------------
def save_ij(buffer):
    normalized = []

    for i in range(1, len(buffer)):
        dx = buffer[i] - buffer[i - 1]
        normalized.append(dx)

    ij_writer.writerow([current_label] + normalized)

def save_uv(buffer):
    dx_values = []
    dy_values = []

    for i in range(1, len(buffer)):
        dx_values.append(buffer[i][0] - buffer[i - 1][0])
        dy_values.append(buffer[i][1] - buffer[i - 1][1])

    scale = max(
        max(abs(v) for v in dx_values),
        max(abs(v) for v in dy_values)
    )

    if scale > 0:
        dx_values = [x / scale for x in dx_values]
        dy_values = [y / scale for y in dy_values]

    features = []
    for dx, dy in zip(dx_values, dy_values):
        features.extend([dx, dy])

    uv_writer.writerow([current_label] + features)

def save_cc(buffer):
    dx_values = []
    dy_values = []

    for i in range(1, len(buffer)):
        dx_values.append(buffer[i][0] - buffer[i - 1][0])
        dy_values.append(buffer[i][1] - buffer[i - 1][1])

    scale = max(
        max(abs(v) for v in dx_values),
        max(abs(v) for v in dy_values)
    )

    if scale > 0:
        dx_values = [x / scale for x in dx_values]
        dy_values = [y / scale for y in dy_values]

    features = []
    for dx, dy in zip(dx_values, dy_values):
        features.extend([dx, dy])

    cc_writer.writerow([current_label] + features)

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

    print(current_label)
    if not current_label == None:
        result = detector.detect(mp_image)

        # -------------
        # Detectar mans
        # -------------
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
 
            wrist_buffer.append(hand[0].x)

            index = hand[8]
            index_buffer.append((index.x, index.y))
            x = int(index.x * frame.shape[1])
            y = int(index.y * frame.shape[0])
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

            thumb = hand[4]
            thumb_buffer.append((thumb.x, thumb.y))
            x = int(thumb.x * frame.shape[1])
            y = int(thumb.y * frame.shape[0])
            cv2.circle(frame, (x, y), 6, (255, 30, 0), -1)

           
    # -----------------
    # Interfaç
    # -----------------
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

    if current_label == "J" or current_label == "I":
        if len(wrist_buffer) == wrist_buffer.maxlen:
            save_ij(wrist_buffer)
            wrist_buffer.clear()
            current_label = None

    elif current_label == "V" or current_label == "U":
        if len(index_buffer) == index_buffer.maxlen:
            save_uv(index_buffer)
            index_buffer.clear()
            current_label = None

    elif current_label == "C" or current_label == "A" or current_label == "Ç":
        if current_label == "A":
            current_label = "Ç"
        if len(thumb_buffer) == thumb_buffer.maxlen:
            save_cc(thumb_buffer)
            thumb_buffer.clear()
            current_label = None

    else:
        current_label = None

cap.release()

uv_file.close()
ij_file.close()
cc_file.close()

cv2.destroyAllWindows()
