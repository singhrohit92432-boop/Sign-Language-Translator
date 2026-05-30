import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load model and preprocessors
model = tf.keras.models.load_model('models/landmark_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load test data
df = pd.read_csv('data/landmark_dataset.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y_enc = le.transform(y)

print("Please run evaluation inside train_improved.py as shown above.")