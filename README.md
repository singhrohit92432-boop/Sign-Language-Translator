# 🤟 Sign-Language-Translator

**Sign Language Translator** turns your webcam into a real‑time American Sign Language interpreter. It recognises finger‑spelled letters, dynamic gestures, and facial emotions.

---

**Overview**

Sign Language Translator is a real‑time sign language interpretation system developed to bridge the communication gap between hearing‑impaired and hearing communities. The system captures hand gestures via a standard webcam, extracts hand landmarks using MediaPipe, and feeds them into two deep learning models: a static letter recognizer (dense/Conv1D) for finger‑spelled words and an LSTM model for dynamic whole‑hand gestures. Additionally, it detects facial emotions (happy, sad, angry, etc.) using DeepFace and reads the final sentence aloud with text‑to‑speech.

The project is designed to be lightweight, real‑time, and user‑friendly. Users can either spell words letter by letter or perform a predefined dynamic gesture – the system builds a sentence in real time. An attractive web dashboard shows the current word, the full sentence, and the user’s facial emotion.

---

**Features**

| What it does | Description |
|--------------|-------------|
| Static spelling | Recognises letters A–Y from hand landmarks. Words are built letter by letter and auto‑corrected. |
| Dynamic gestures | Detects whole‑hand movements using an LSTM model (hello, thank you, stop, love, please, my). |
| Facial emotion | Analyses your face in real time (happy, sad, angry, etc.). |
| Text‑to‑speech | Reads the final sentence aloud with one click. |
| Auto‑correction | Fixes common spelling mistakes (e.g., "HELH" → "HELLO"). |
| Modern UI | Glass‑morphism design, responsive, works on any device. |

---

**How It Works**

1. MediaPipe extracts 21 hand landmarks (x, y, z) from each video frame.
2. Two models run in parallel: a dense/Conv1D model recognises static letters, and an LSTM model recognises dynamic gestures.
3. DeepFace analyses your face to detect emotions.
4. Words are auto‑corrected (TextBlob + custom dictionary) and assembled into a sentence.
5. Flask serves a web interface that shows everything in real time.

---

**Tech Stack**

| Layer | Technology |
|-------|------------|
| Backend | Flask (Python) |
| Computer Vision | OpenCV, MediaPipe |
| Deep Learning | TensorFlow / Keras |
| Emotion AI | DeepFace |
| NLP | TextBlob |
| Frontend | HTML, Tailwind CSS, JavaScript |

---

**Usage**

| Action | Result |
|--------|--------|
| Spell a word (e.g., H‑E‑L‑L‑O) | Hold each letter steady; raw word appears. Lower hand → word is corrected and added to sentence. |
| Perform a dynamic gesture (wave, heart, etc.) | The word is instantly added to the sentence. |
| Look at the camera | Your facial emotion is displayed. |
| Click "Speak Sentence" | The sentence is read aloud. |
| Click "Reset All" | Clears the sentence and current word. |

---

**Train Your Own Models**
```bash
Static letters (A–Y)
python collect_landmarks.py
python train_improved.py
```
**Dynamic gestures**
```bash
python collect_motion.py
python train_lstm_motion.py
```
---
**Project Structure**
```
signlanguage/
|├── sign.py                     # Main Flask application|
├── collect_landmarks.py        # Collect static letter data (CSV)
├── collect_motion.py           # Collect dynamic gesture sequences
├── train_improved.py           # Train static landmark model
├── train_lstm_motion.py        # Train LSTM for dynamic gestures
├── models/
│   ├── landmark_model.h5
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   ├── lstm_dynamic_model.h5
│   ├── lstm_dynamic_label_encoder.pkl
│   └── lstm_dynamic_scaler.pkl
├── data/
│   ├── landmark_dataset.csv
│   └── dynamic_data/
├── requirements.txt
└── README.md
```
---

**Future Improvements**
- Support two‑handed signs.
- Add more dynamic gestures (yes, no, help, friend, etc.).
- Improve sentence‑level sentiment analysis.
- Add database support for user history.

---

Authors

Rohit Singh


B.Tech (2nd Year)

⭐ If you found this project useful, give it a star ⭐
---

