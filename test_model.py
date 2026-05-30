import numpy as np
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('models/landmark_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


dummy_v = np.random.randn(63).reshape(1, -1)
dummy_q = np.random.randn(63).reshape(1, -1)

pred_v = model.predict(scaler.transform(dummy_v))
pred_q = model.predict(scaler.transform(dummy_q))

print("Classes in order:", le.classes_)
print("Prediction for dummy V:", le.classes_[np.argmax(pred_v)])
print("Prediction for dummy Q:", le.classes_[np.argmax(pred_q)])