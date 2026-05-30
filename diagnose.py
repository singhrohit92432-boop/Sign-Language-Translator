import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

#Checking data
df = pd.read_csv('data/landmark_dataset.csv')
print("=== DATA DISTRIBUTION ===")
print(df['label'].value_counts())
print(f"Total samples: {len(df)}")

#Load model
model = tf.keras.models.load_model('models/landmark_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"\n=== MODEL CLASSES ===")
print(le.classes_)

print("\n=== SAMPLE PREDICTIONS ===")
for letter in le.classes_:
    sample = df[df['label'] == letter].iloc[0, :-1].values.reshape(1, -1)
    scaled = scaler.transform(sample)
    pred = model.predict(scaled, verbose=0)[0]
    idx = np.argmax(pred)
    print(f"True: {letter} -> Predicted: {le.classes_[idx]} (conf: {pred[idx]:.2f})")