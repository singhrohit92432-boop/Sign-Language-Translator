from flask import Flask, render_template_string, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import time
import threading

app = Flask(__name__)

# Load LSTM model
print("Loading dynamic LSTM model...")
model = tf.keras.models.load_model('models/lstm_dynamic_model.h5')
with open('models/lstm_dynamic_label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/lstm_dynamic_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Classes:", le.classes_)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Shared state
state = {"prediction": "", "confidence": 0.0}
lock = threading.Lock()

# Sequence buffer
sequence = deque(maxlen=30)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignLens Motion – Dynamic Gestures</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: radial-gradient(circle at 20% 30%, #0f172a, #030712); }
        .glass { backdrop-filter: blur(12px); background: rgba(15,23,42,0.5); border: 1px solid rgba(59,130,246,0.3); }
    </style>
</head>
<body class="min-h-screen p-6">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-5xl font-bold text-white text-center mb-6">🤟 SignLens <span class="text-purple-400">Motion</span></h1>
        <p class="text-center text-slate-400 mb-8">Recognises dynamic gestures: hello, thank you, stop, love, please, my</p>
        <div class="glass rounded-2xl p-4">
            <img src="/video_feed" id="video" class="w-full rounded-xl">
        </div>
        <div class="glass rounded-2xl p-6 mt-6 text-center">
            <div class="text-slate-400 text-sm uppercase">Recognised Gesture</div>
            <div class="text-6xl font-bold text-purple-400 break-words mt-2" id="gesture">—</div>
            <div class="text-slate-500 text-xs mt-2">Confidence: <span id="confidence">0%</span></div>
        </div>
        <div class="text-center text-slate-500 text-xs mt-6">
            Perform a dynamic sign (e.g., wave hello, hand to heart for love)
        </div>
    </div>
    <script>
        async function fetchState() {
            const res = await fetch('/api/state');
            const data = await res.json();
            document.getElementById('gesture').innerText = data.prediction || '—';
            document.getElementById('confidence').innerText = `${(data.confidence * 100).toFixed(0)}%`;
        }
        setInterval(fetchState, 200);
        fetchState();
    </script>
</body>
</html>
"""

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints)
    return np.zeros(63)

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        
        # Extract keypoints and update sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        
        # Predict every 30 frames
        if len(sequence) == 30:
            seq_array = np.array(sequence).reshape(1, 30, 63)
            # Scale
            seq_flat = seq_array.reshape(-1, 63)
            seq_scaled = scaler.transform(seq_flat).reshape(1, 30, 63)
            pred = model.predict(seq_scaled, verbose=0)[0]
            idx = np.argmax(pred)
            conf = pred[idx]
            if conf > 0.7:
                gesture = le.classes_[idx].replace('_', ' ')
                with lock:
                    state["prediction"] = gesture
                    state["confidence"] = float(conf)
                print(f"Gesture: {gesture} ({conf:.2f})")
            # Clear sequence to prepare for next gesture
            sequence.clear()
        
        # Draw status on frame
        with lock:
            current = state["prediction"]
        cv2.putText(frame, f"Dynamic: {current}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    with lock:
        return jsonify(state)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)