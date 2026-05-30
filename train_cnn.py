import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

print("Starting CNN training...")
os.makedirs('models', exist_ok=True)

print("Loading data...")
train_df = pd.read_csv('data/sign_mnist_train.csv')
test_df = pd.read_csv('data/sign_mnist_test.csv')
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


y_train = train_df.iloc[:,0].values
y_test = test_df.iloc[:,0].values
print(f"Label range train: {y_train.min()} to {y_train.max()}")
print(f"Label range test: {y_test.min()} to {y_test.max()}")

X_train = train_df.iloc[:,1:].values.reshape(-1, 28, 28, 1) / 255.0
X_test = test_df.iloc[:,1:].values.reshape(-1, 28, 28, 1) / 255.0

#CNN 
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(25, activation='softmax')   
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training...")
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=64)
model.save('models/cnn_asl_model.h5')
print("✅ CNN model saved in models/cnn_asl_model.h5")
