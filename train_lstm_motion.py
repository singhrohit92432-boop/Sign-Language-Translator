import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = 'dynamic_data'
if not os.path.exists(DATA_DIR):
    print("No dynamic data found. Run collect_dynamic.py first.")
    exit()

# Load all sequences
sequences = []
labels = []
for gesture in os.listdir(DATA_DIR):
    gesture_path = os.path.join(DATA_DIR, gesture)
    if not os.path.isdir(gesture_path):
        continue
    for file in os.listdir(gesture_path):
        if file.endswith('.npy'):
            seq = np.load(os.path.join(gesture_path, file))
            sequences.append(seq)
            labels.append(gesture)

if not sequences:
    print("No data loaded. Check your dynamic_data folder.")
    exit()

X = np.array(sequences)   # shape: (samples, 30, 63)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_enc)

# Scale each frame
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, 63)
X_test_flat = X_test.reshape(-1, 63)
scaler.fit(X_train_flat)
X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# Build LSTM model
model = Sequential([
    Masking(mask_value=0., input_shape=(30, 63)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

print("Training LSTM on dynamic gestures...")
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=16,
                    validation_data=(X_test_scaled, y_test), callbacks=[early_stop], verbose=1)

loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.2f}%")

# Save model and preprocessors
os.makedirs('models', exist_ok=True)
model.save('models/lstm_dynamic_model.h5')
with open('models/lstm_dynamic_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('models/lstm_dynamic_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Model saved to 'models/lstm_dynamic_model.h5'")