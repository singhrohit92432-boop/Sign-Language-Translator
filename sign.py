from flask import Flask, render_template_string, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import time
import threading
import os
from deepface import DeepFace
from textblob import Word

app = Flask(__name__)


print("Loading ASL landmark model...")
asl_model = tf.keras.models.load_model('models/landmark_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("ASL classes:", list(le.classes_))


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


state = {
    "sentence": "",
    "current_word": "",
    "emotion_face": {"label": "neutral", "emoji": "😐"}
}
lock = threading.Lock()


def correct_word(raw_word):
    if len(raw_word) < 2:
        return raw_word.upper()
    
    custom = {
        "HELH": "HELLO", "HELLOC": "HELLO", "HEL": "HELLO",
        "HEM": "HELLO", "HELO": "HELLO", "HLO": "HELLO",
        "THNK": "THANK", "THNX": "THANKS", "PLS": "PLEASE",
        "FRND": "FRIEND", "LOV": "LOVE", "HAPY": "HAPPY",
        "SAD": "SAD", "ANGRY": "ANGRY", "AE": "AM",
        "U": "YOU", "R": "ARE","Y": "WHY", "W": "WHAT","SY":"MY","WANEU":"WATER","GOEU":"GO","GIVEU":"GIVE","LOVM":"LOVE","FASILY": "FAMILY","SATORU":"SATORU","SET":"EAT","TOMOROW":"TOMORROW","LOVS":"LOVE"
    }
    lower_raw = raw_word.lower()
    if lower_raw in custom:
        return custom[lower_raw].upper()
    
    try:
        blob_word = Word(lower_raw)
        corrected = blob_word.correct().upper()
        
        if len(corrected) < 2 or corrected == raw_word.upper():
            return raw_word.upper()
        return corrected
    except:
        return raw_word.upper()


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignLens Pro – ASL Interpreter</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Inter', sans-serif; }
        body {
            background: radial-gradient(ellipse at top, #0f172a, #020617);
            min-height: 100vh;
        }
        .glow-card {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(59,130,246,0.4);
            transition: all 0.3s ease;
        }
        .glow-card:hover {
            border-color: rgba(59,130,246,0.8);
            box-shadow: 0 0 20px rgba(59,130,246,0.2);
        }
        .word-glow {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { text-shadow: 0 0 5px #3b82f6; }
            100% { text-shadow: 0 0 20px #3b82f6; }
        }
        .emotion-badge {
            font-size: 2.5rem;
            transition: transform 0.2s;
        }
        .emotion-badge:hover { transform: scale(1.1); }
    </style>
</head>
<body class="p-6">
    <div class="max-w-6xl mx-auto">
        <div class="text-center mb-12">
            <h1 class="text-6xl font-extrabold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                 Sign Language Interpreter <span class="text-white">Pro</span>
            </h1>
            <p class="text-slate-400 mt-2">Real‑time ASL with auto‑correction</p>
        </div>

        <div class="grid md:grid-cols-2 gap-8">
            <div class="glow-card rounded-2xl p-4 shadow-2xl">
                <div class="relative rounded-xl overflow-hidden bg-black">
                    <img src="/video_feed" id="video" class="w-full">
                </div>
                <p class="text-slate-400 text-center text-sm mt-3">✋ Hold a sign – raw letters appear. Lower hand to finalise word.</p>
            </div>

            <div class="space-y-6">
                <div class="glow-card rounded-2xl p-6">
                    <div class="text-slate-400 text-sm uppercase tracking-wider">Current Word (raw)</div>
                    <div class="text-5xl font-bold text-blue-400 break-words word-glow mt-2" id="current_word">—</div>
                </div>

                <div class="glow-card rounded-2xl p-6">
                    <div class="text-slate-400 text-sm uppercase tracking-wider">Full Sentence (corrected)</div>
                    <div class="text-slate-200 text-xl leading-relaxed break-words max-h-48 overflow-y-auto mt-2" id="sentence"></div>
                </div>

                <div class="glow-card rounded-2xl p-6 text-center">
                    <div class="text-slate-400 text-sm uppercase tracking-wider mb-2">🎭 Live Face Emotion</div>
                    <div class="emotion-badge" id="face_emotion">😐 neutral</div>
                </div>

                <div class="glow-card rounded-2xl p-6">
                    <div class="flex gap-4">
                        <button id="speakBtn" class="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white py-3 rounded-xl text-lg transition shadow-lg">
                            🔊 Speak Sentence
                        </button>
                        <button id="resetBtn" class="flex-1 bg-rose-600 hover:bg-rose-700 text-white py-3 rounded-xl text-lg transition shadow-lg">
                            🗑️ Reset All
                        </button>
                    </div>
                </div>
            </div>
        </div>
        <footer class="text-center text-slate-500 text-xs mt-12">
            Powered by MediaPipe • TensorFlow • DeepFace • Auto‑correction
        </footer>
    </div>

    <script>
        async function fetchState() {
            const res = await fetch('/api/state');
            const data = await res.json();
            document.getElementById('current_word').innerText = data.current_word || '—';
            document.getElementById('sentence').innerText = data.sentence || '';
            if (data.emotion_face) {
                document.getElementById('face_emotion').innerHTML = `${data.emotion_face.emoji} ${data.emotion_face.label}`;
            }
        }

        async function resetAll() {
            await fetch('/reset', {method: 'POST'});
            fetchState();
        }

        function speak() {
            const text = document.getElementById('sentence').innerText;
            if (text && text !== '—') {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 0.9;
                utterance.pitch = 1.0;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            } else {
                alert("No sentence yet – sign something first!");
            }
        }

        document.getElementById('resetBtn').onclick = resetAll;
        document.getElementById('speakBtn').onclick = speak;

        setInterval(fetchState, 200);
        fetchState();
    </script>
</body>
</html>
"""


DEBOUNCE_FRAMES = 2          
CONFIDENCE_THRESH = 0.15
PAUSE_FRAMES = 20          
FRAME_SKIP = 0               

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    letter_buffer = deque()
    raw_word = ""             
    sentence = ""
    no_hand_counter = 0
    last_letter = ""
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_counter += 1

        if FRAME_SKIP > 0 and frame_counter % (FRAME_SKIP + 1) != 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            no_hand_counter = 0
            hand = results.multi_hand_landmarks[0]
            lm = []
            for p in hand.landmark:
                lm.extend([p.x, p.y, p.z])
            inp = np.array(lm).reshape(1, -1)
            inp_scaled = scaler.transform(inp)
            pred = asl_model.predict(inp_scaled, verbose=0)[0]
            idx = np.argmax(pred)
            conf = pred[idx]
            letter = le.classes_[idx] if conf > CONFIDENCE_THRESH else None

            if letter:
                letter_buffer.append(letter)
                while len(letter_buffer) > DEBOUNCE_FRAMES:
                    letter_buffer.popleft()
                if len(letter_buffer) == DEBOUNCE_FRAMES and len(set(letter_buffer)) == 1:
                    stable = letter_buffer[0]
                    if stable != last_letter:
                        raw_word += stable
                        last_letter = stable
                        with lock:
                            state["current_word"] = raw_word
                        print(f"➕ {stable} -> raw: {raw_word}")
                        letter_buffer.clear()
            else:
                letter_buffer.clear()
                last_letter = ""
        else:
            no_hand_counter += 1
            if no_hand_counter >= PAUSE_FRAMES and raw_word:
                # Finalise the word then correct it and add to sentence
                corrected = correct_word(raw_word)
                sentence = (sentence + " " + corrected).strip()
                raw_word = ""
                with lock:
                    state["sentence"] = sentence
                    state["current_word"] = ""
                print(f"📝 Added corrected word: '{corrected}' | Full: {sentence}")
                no_hand_counter = 0
            letter_buffer.clear()
            last_letter = ""

        # Face emotion every 10 frames
        if frame_counter % 10 == 0:
            try:
                temp_path = "temp_face.jpg"
                cv2.imwrite(temp_path, frame)
                result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                os.remove(temp_path)
                if result and isinstance(result, list):
                    face_label = result[0]['dominant_emotion']
                else:
                    face_label = result['dominant_emotion']
                emoji_map = {'happy':'😊','sad':'😢','angry':'😠','surprise':'😲','fear':'😨','disgust':'🤢','neutral':'😐'}
                face_emotion = {"label": face_label, "emoji": emoji_map.get(face_label, '😐')}
                with lock:
                    state["emotion_face"] = face_emotion
            except:
                pass

        
        cv2.putText(frame, f"Word: {raw_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

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

@app.route('/reset', methods=['POST'])
def reset():
    with lock:
        state["sentence"] = ""
        state["current_word"] = ""
        state["emotion_face"] = {"label": "neutral", "emoji": "😐"}
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)