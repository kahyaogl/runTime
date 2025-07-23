# train_lstm.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Eğitim parametreleri ---
window_size = 50
epochs = 5
batch_size = 32

# --- Veri yükle ve normalleştir ---
df = pd.read_csv("normaliler.csv", header=None, usecols=[0])
df.columns = ['value']

scaler = MinMaxScaler()
df['scaled'] = scaler.fit_transform(df[['value']])

def create_sequences(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

X = create_sequences(df['scaled'].values, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Model tanımı ---
inputs = Input(shape=(window_size, 1))
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(window_size)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=2)

# --- MSE, ortalama ve std hesapla ---
X_pred = autoencoder.predict(X)
mse = np.mean(np.mean(np.square(X - X_pred), axis=1), axis=1)
mean_mse = np.mean(mse)
std_mse = np.std(mse)

# --- Model ve parametreleri kaydet ---
autoencoder.save("lstm_autoencoder_model.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(mean_mse, "mean_mse.pkl")
joblib.dump(std_mse, "std_mse.pkl")

print("✅ Model eğitildi ve kaydedildi.")