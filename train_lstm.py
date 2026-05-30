import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

print("Starting LSTM training...")
os.makedirs('models', exist_ok=True)


np.random.seed(42)
X_dummy = np.random.rand(100, 30, 63)
y_dummy = np.array(['hello']*20 + ['thanks']*20 + ['yes']*20 + ['no']*20 + ['iloveyou']*20)

print(f"Data shape: {X_dummy.shape}")
print(f"Labels: {np.unique(y_dummy)}")


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_dummy)
print(f"Encoded classes: {label_encoder.classes_}")


scaler = StandardScaler()
X_flat = X_dummy.reshape(-1, 63)
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(100, 30, 63)

#LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 63)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training LSTM on dummy data...")
model.fit(X_scaled, y_encoded, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

model.save('models/lstm_model.h5')
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ LSTM model saved in 'models/lstm_model.h5'")
print("✅ label_encoder.pkl saved")
print("✅ scaler.pkl saved")
