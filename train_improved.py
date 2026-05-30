import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data
df = pd.read_csv('data/landmark_dataset.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("Class distribution before augmentation:")
print(df['label'].value_counts())


le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_enc)


X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_enc)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def augment_landmarks(X, y):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        X_aug.append(X[i]); y_aug.append(y[i])
        for _ in range(8):   # 8 augmented copies per original
            noise = np.random.normal(0, 0.02, X[i].shape)
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.05, 0.05, X[i].shape)
            
            angle = np.random.uniform(-0.2, 0.2)
            cos, sin = np.cos(angle), np.sin(angle)
            rotated = X[i].copy()
            for j in range(0, 63, 3):
                x_, y_ = rotated[j], rotated[j+1]
                rotated[j] = x_ * cos - y_ * sin
                rotated[j+1] = x_ * sin + y_ * cos
            aug = rotated * scale + noise + shift
            X_aug.append(aug); y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)

X_train_aug, y_train_aug = augment_landmarks(X_train, y_train)
print(f"Original train: {X_train.shape}, Augmented: {X_train_aug.shape}")


class_weights = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)
class_weight_dict = dict(enumerate(class_weights))


model = Sequential([
    Reshape((21, 3), input_shape=(63,)),
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)


loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test accuracy: {acc*100:.2f}%")

#Confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('models/confusion_matrix.png')
print("Confusion matrix saved to models/confusion_matrix.png")


os.makedirs('models', exist_ok=True)
model.save('models/landmark_model.h5')
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Model and preprocessors saved to 'models/'")