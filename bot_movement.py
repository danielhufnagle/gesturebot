import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import pyfirmata

port = '/dev/cu.usbmodem1301'
board = pyfirmata.Arduino(port)

# setup for mediapipe detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands # hand model
mp_holistic = mp.solutions.holistic

def detection(image, model):
    # handles detection of hands and hand landmark positions
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # draws landmarks and connecting lines onto the video feed
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(120, 0, 200), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=4)
    )

def extract_keypoints(results):
    # gets the coordinate information of the landmarks
    hand_points = np.array([[result.x, result.y, result.z] for result in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return hand_points

# list of all recognizable gestures
actions = np.array(['thumbs_up', 'thumbs_down', 'open', 'fist', 'peace', 'point', 'middle_finger', 'rock', 'stop', 'okay', 'call_me', 'none'])

# get model
model = tf.keras.models.load_model('model.h5')
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# record and classify gestures
vid = cv2.VideoCapture(0)
action = 'call_me' # replace with whatever gesture you want to collect
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while vid.isOpened():
        _, image = vid.read()
        image, results = detection(image, holistic)
        draw_landmarks(image, results)
        keypoints = extract_keypoints(results)
        keypoints = np.array([keypoints])
        prediction = probability_model.predict(keypoints)
        image = cv2.flip(image, 1)
        cv2.putText(image, actions[np.argmax(prediction)], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 44, 250), 2, cv2.LINE_AA)
        cv2.imshow('Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()
# no idea why the hell this is needed, but the capture window won't close otherwise
for i in range(4):
    cv2.waitKey(1)