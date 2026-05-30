import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import Counter

print("Loading model...")
model = tf.keras.models.load_model('models/landmark_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print(f"Ready. Classes: {le.classes_}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

prediction_buffer = []
BUFFER_SIZE = 5

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = []
        for p in hand.landmark:
            lm.extend([p.x, p.y, p.z])
        inp = np.array(lm).reshape(1, -1)
        inp_scaled = scaler.transform(inp)
        pred = model.predict(inp_scaled, verbose=0)[0]
        idx = np.argmax(pred)
        conf = pred[idx]
        letter = le.classes_[idx] if conf > 0.35 else "?"
        # Majority vote buffer
        if letter != "?":
            prediction_buffer.append(letter)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer.pop(0)
            if len(prediction_buffer) == BUFFER_SIZE:
                counter = Counter(prediction_buffer)
                most_common, count = counter.most_common(1)[0]
                if count >= 3:
                    display_letter = most_common
                else:
                    display_letter = "?"
            else:
                display_letter = "?"
        else:
            prediction_buffer.clear()
            display_letter = "?"
        cv2.putText(frame, f"{display_letter} ({conf:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Live Test - press ESC to quit", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import Counter
import threading
import time
import requests   #to send updates to Flask


print("Loading model...")
model = tf.keras.models.load_model('models/landmark_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Classes:", list(le.classes_))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


current_word = ""
sentence = ""

def send_update():
    """Send current_word and sentence to Flask server (runs every 200ms)"""
    try:
        requests.post('http://localhost:5001/update', json={'word': current_word, 'sentence': sentence}, timeout=0.05)
    except:
        pass


def main():
    global current_word, sentence
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prediction_buffer = []
    BUFFER_SIZE = 5
    MAJORITY_COUNT = 3
    no_hand_counter = 0
    PAUSE_FRAMES = 30

    last_update_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            no_hand_counter = 0
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                lm = []
                for p in hand.landmark:
                    lm.extend([p.x, p.y, p.z])
                inp = np.array(lm).reshape(1, -1)
                inp_scaled = scaler.transform(inp)
                pred = model.predict(inp_scaled, verbose=0)[0]
                idx = np.argmax(pred)
                conf = pred[idx]
                letter = le.classes_[idx] if conf > 0.25 else "?"
                print(f"DEBUG: {letter} ({conf:.2f})")

                if letter != "?":
                    prediction_buffer.append(letter)
                    if len(prediction_buffer) > BUFFER_SIZE:
                        prediction_buffer.pop(0)
                    if len(prediction_buffer) == BUFFER_SIZE:
                        counter = Counter(prediction_buffer)
                        most_common, count = counter.most_common(1)[0]
                        if count >= MAJORITY_COUNT:
                            current_word += most_common
                            print(f"✅ ADDED: {most_common} -> {current_word}")
                            prediction_buffer.clear()
                else:
                    prediction_buffer.clear()
        else:
            no_hand_counter += 1
            if no_hand_counter >= PAUSE_FRAMES and current_word:
                sentence = (sentence + " " + current_word).strip()
                print(f"📝 Sentence: {sentence}")
                current_word = ""
                no_hand_counter = 0
            prediction_buffer.clear()

       
        cv2.putText(frame, f"Word: {current_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, f"Sentence: {sentence[-50:]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("SignLens - Live ASL (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Send update to Flask every 200ms 
        if time.time() - last_update_time > 0.2:
            send_update()
            last_update_time = time.time()
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()