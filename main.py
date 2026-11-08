# Importar les llibreries
import pandas as pd
import joblib
from collections import deque
from sklearn.neural_network import MLPClassifier

import os
import sys

import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# Definir la ubicacio de les carpetes relatiu a l'executable
def resource_path(relative_path):
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Crear les opcions per al model de MediaPipe
base_options = python.BaseOptions(model_asset_path=resource_path("models/hand_landmarker.task"))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

# Importar el model de MediaPipe a partir de les opcions
detector = vision.HandLandmarker.create_from_options(options)

# Obrir la camera
cap = cv2.VideoCapture(0)

# Crear la finestra
cv2.namedWindow(
    "LSC",
    cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
)

# Carregar el model de LSC
model = joblib.load(resource_path("models/lsc.pkl"))
ij_model = joblib.load(resource_path("models/ij.pkl"))
uv_model = joblib.load(resource_path("models/uv.pkl"))

ij_predictions = deque(maxlen=10)
wrist_buffer = deque(maxlen=20)

uv_predictions = deque(maxlen=10)
index_buffer = deque(maxlen=20)

prediction = ["..."]

while True:
    # Agafar la foto de la camera
    ret, frame = cap.read()
    h, w, _ = frame.shape

    # Convertir la frama a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convertir la imatge RGB a una imatge de MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Executar la deteccio
    result = detector.detect(mp_image)

    # Extreure els punts de la ma si se'n detecta
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        raw_landmarks = []

        # Dibuixar els punts
        for lm in hand:
            cx = int(lm.x * w)
            cy = int(lm.y * h)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            raw_landmarks.append((lm.x, lm.y, lm.z))
        
        wrist_buffer.append(raw_landmarks[0][0])
        index_buffer.append((raw_landmarks[8][0], raw_landmarks[8][1]))
        
        # Normalitzar les coordenades
        wrist_x, wrist_y, wrist_z = raw_landmarks[0]

        normalized = []

        for x, y, z in raw_landmarks:
            normalized.extend([
                x - wrist_x,
                y - wrist_y,
                z - wrist_z
            ])

        # Fer la predicció
        prediction = model.predict([normalized])

        # Comprovar en cas de lletres amb moviment
        if prediction == "I":
            prediction = "..."
            if len(wrist_buffer) == wrist_buffer.maxlen:
                normalized = []
                for i in range(1, len(wrist_buffer)):
                    dx = wrist_buffer[i] - wrist_buffer[i - 1]
                    normalized.append(dx)

                prediction = ij_model.predict([normalized])
                ij_predictions.append(prediction)

                if len(ij_predictions) == ij_predictions.maxlen:
                    count = ij_predictions.count("J")
                    if count > 0:
                        prediction = "J"
                    else:
                        prediction = "I"

        elif prediction == "U":
            prediction = "..."
            if len(index_buffer) == index_buffer.maxlen:
                normalized = []

                for i in range(1, len(index_buffer)):
                    dx = index_buffer[i][0] - index_buffer[i - 1][0]
                    dy = index_buffer[i][1] - index_buffer[i - 1][1]
                    normalized.append(dx)
                    normalized.append(dy)

                prediction = uv_model.predict([normalized])
                uv_predictions.append(prediction)

                if len(uv_predictions) == uv_predictions.maxlen:
                    count = uv_predictions.count("V")

                    if count > 0:
                        prediction = "V"
                    else:
                        prediction = "U"

        # Escriure la lletra que ha predit el model a la pantalla
        cv2.putText(
            frame,
            f"PREDICCIO: {prediction[0]}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Mostrar la finestra
    cv2.imshow("LSC", frame)

    key = cv2.waitKey(1) & 0xFF
    # Si es prem la tecla 1 sortir
    if key == ord('1'):
        break

# Apagar la càmera i tancar la finestra
cap.release()
cv2.destroyAllWindows()
