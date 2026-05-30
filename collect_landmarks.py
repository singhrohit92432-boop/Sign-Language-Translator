import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
os.makedirs('data', exist_ok=True)
data = []
letter = None
frame_count = 0
samples = 200          
print("Collect 200 frames per letter. Vary distance, angle, lighting.")
print("Press letter key (A-Y, no J). Hold sign steady. ESC to quit.")

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

        if letter and frame_count < samples:
            data.append(lm + [letter])
            frame_count += 1
            cv2.putText(frame, f"Recording {letter}: {frame_count}/{samples}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            if frame_count % 50 == 0 and frame_count > 0:
                print(f"  → Change hand distance or angle slightly for better variation")
        else:
            cv2.putText(frame, f"Press key for letter (A-Y). Last: {letter if letter else 'None'}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    else:
        cv2.putText(frame, "Show your hand to camera", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Collect Landmarks - vary hand pose", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif 97 <= key <= 121:
        ch = chr(key).upper()
        if ch != 'J':
            letter = ch
            frame_count = 0
            print(f"\nRecording '{letter}' – hold still, but vary position gradually.")

cap.release()
cv2.destroyAllWindows()

if data:
    
    file_exists = os.path.isfile('data/landmark_dataset.csv')
    with open('data/landmark_dataset.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([f'lm_{i}' for i in range(63)] + ['label'])
        writer.writerows(data)
    print(f"✅ Appended {len(data)} frames to data/landmark_dataset.csv")
else:
    print("No data collected.")