import pandas as pd

import joblib

from sklearn.neural_network import MLPClassifier

import csv
import cv2
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# MEDIAPIPE
base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)
# ------------------------
 
# Webcam
capture = cv2.VideoCapture(0)

capture.set(3, 640)
capture.set(4, 480)
# ------------------------

# CSV
cc_file = open("data/cc.csv", "a", newline="")
cc_writer = csv.writer(cc_file)

cc_data = pd.read_csv("data/cc.csv", header=None)

ccX = cc_data.iloc[:, 1:]
ccy = cc_data.iloc[:, 0]

cc_model = MLPClassifier(
    hidden_layer_sizes=(32,),
    max_iter=10000
)

print("Entrenant...")
cc_model.fit(ccX, ccy)
print("llest!")

predictions = deque(maxlen=10)

normalized = []
thumb_buffer = deque(maxlen=20)

# Loop
while True:
    ret, frame = capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data = rgb_frame
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
    
        landmarks = []
        for lm in hand:
            landmarks.append((lm.x, lm.y, lm.z))

        thumb_buffer.append((landmarks[4][0], landmarks[4][1]))

        if len(thumb_buffer) == thumb_buffer.maxlen:
            normalized = []

            for i in range(1, len(thumb_buffer)):
                dx = thumb_buffer[i][0] - thumb_buffer[i-1][0]
                dy = thumb_buffer[i][1] - thumb_buffer[i-1][0]
                normalized.append(dx)
                normalized.append(dy)

            prediction = cc_model.predict([normalized])
            print(prediction)


    cv2.imshow("cc", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        joblib.dump(cc_model, "models/cc.pkl")
        print("Saved Model")

cv2.destroyAllWindows()
capture.release()
