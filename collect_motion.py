import cv2
import mediapipe as mp
import numpy as np
import os
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Define gestures to collect
GESTURES = ['hello', 'thank_you', 'stop', 'love', 'please', 'my']
SEQUENCE_LENGTH = 30
SAMPLES_PER_GESTURE = 50   # Increase for better accuracy

os.makedirs('dynamic_data', exist_ok=True)

print("Dynamic Gesture Data Collection")
print("================================")
print(f"Collecting {SAMPLES_PER_GESTURE} sequences per gesture")
print("Press SPACE to start recording. Press ESC to skip gesture.\n")

for gesture in GESTURES:
    print(f"\n--- Prepare to record '{gesture}' ---")
    input("Press ENTER when ready...")
    gesture_dir = os.path.join('dynamic_data', gesture)
    os.makedirs(gesture_dir, exist_ok=True)
    
    sample_count = len(os.listdir(gesture_dir))
    while sample_count < SAMPLES_PER_GESTURE:
        print(f"\nRecording sample {sample_count+1}/{SAMPLES_PER_GESTURE} for '{gesture}'")
        print("Perform the gesture now... (press SPACE when done, ESC to cancel)")
        
        sequence = []
        recording = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            keypoints = np.zeros(63)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                for i, lm in enumerate(hand.landmark):
                    keypoints[i*3] = lm.x
                    keypoints[i*3+1] = lm.y
                    keypoints[i*3+2] = lm.z
            
            if recording:
                sequence.append(keypoints)
                cv2.putText(frame, f"Recording {len(sequence)}/{SEQUENCE_LENGTH}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(frame, "Press SPACE to start recording", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.imshow("Record Dynamic Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):   # space
                if not recording:
                    recording = True
                    sequence = []
                    print("Recording started...")
                else:
                    # Stop recording, save if we have enough frames
                    if len(sequence) >= SEQUENCE_LENGTH:
                        # Trim or pad to exactly 30 frames
                        sequence = sequence[:SEQUENCE_LENGTH]
                        while len(sequence) < SEQUENCE_LENGTH:
                            sequence.append(np.zeros(63))
                        np.save(os.path.join(gesture_dir, f"{sample_count}.npy"), sequence)
                        sample_count += 1
                        print(f"Saved sample {sample_count}/{SAMPLES_PER_GESTURE}")
                    else:
                        print(f"Not enough frames ({len(sequence)}). Need {SEQUENCE_LENGTH}. Discarded.")
                    recording = False
                    break
            elif key == 27:   # ESC
                print("Skipped this gesture")
                break
        
        if key == 27:   # ESC pressed during gesture
            break
    
    print(f"Finished collecting '{gesture}'")

cap.release()
cv2.destroyAllWindows()
print("\nData collection complete! Files saved in 'dynamic_data/'")