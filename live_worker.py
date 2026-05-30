import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import Counter
import time
import json

#loading model
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


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✅ Camera opened. Press 'q' in video window to quit.")

    current_word = ""
    sentence = ""
    prediction_buffer = []
    BUFFER_SIZE = 5
    MAJORITY_COUNT = 3
    no_hand_counter = 0
    PAUSE_FRAMES = 30

    last_save_time = time.time()

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

        
        if time.time() - last_save_time >= 0.2:
            with open('sign_data.json', 'w') as f:
                json.dump({'word': current_word, 'sentence': sentence}, f)
            last_save_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()