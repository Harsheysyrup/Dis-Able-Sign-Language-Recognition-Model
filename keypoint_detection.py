import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands  # hands model
mp_drawing = mp.solutions.drawing_utils  # drawing utilities


def mediapipe_detection(image, model):
    # MediaPipe expects RGB images and performs best when the input buffer is
    # marked read-only during processing.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    # Overlay MediaPipe's default hand skeleton on the frame.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def extract_keypoints(results):
    # Flatten 21 hand landmarks into the shape expected by the LSTM:
    # 21 landmarks * (x, y, z) = 63 values.
    handlandmark = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            handlandmark = np.array([[res.x, res.y, res.z] for res in
                                     i.landmark]).flatten()

    return np.concatenate([handlandmark])
