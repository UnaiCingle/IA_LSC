import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# Carregar model
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

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
    i = 0
    if result.hand_landmarks:
        for hand in result.hand_landmarks:

            landmarks = []
            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])

                # Draw
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 80 * i, 0), -1)
            
            i += 1

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()