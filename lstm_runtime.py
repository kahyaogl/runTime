import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import serial
import time
from collections import deque

# --- 1. Eğitim parametreleri ---
window_size = 30
epochs = 5
batch_size = 32

# --- 2. Veriyi yükle ve normalleştir ---
df = pd.read_csv("sadece_normal_veri.csv", header=None, usecols=[0])
df.columns = ['value']

scaler = MinMaxScaler()
df['scaled'] = scaler.fit_transform(df[['value']])

def create_sequences(data, window_size=30):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

X = create_sequences(df['scaled'].values, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- 3. Model tanımla ---
inputs = Input(shape=(window_size, 1))
encoded = LSTM(128, activation='relu')(inputs)
decoded = RepeatVector(window_size)(encoded)
decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=2)

# --- 4. Eğitim sonrası mse, mean ve std hesapla ---
X_pred = autoencoder.predict(X)
mse = np.mean(np.mean(np.square(X - X_pred), axis=1), axis=1)

mean_mse = np.mean(mse)
std_mse = np.std(mse)

# --- 5. Model ve scaler dosyalarını kaydet ---
autoencoder.save("lstm_autoencoder_model.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(mean_mse, "mean_mse.pkl")
joblib.dump(std_mse, "std_mse.pkl")

# --- 6. Gerçek zamanlı anomali tespiti için hazırlık ---
port = "COM4"
baud_rate = 230400
threshold_z = 3

# Kaydırmalı pencere
data_window = deque(maxlen=window_size)

# Gerçek zamanlı grafik için pencere ve değer listeleri
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))

time_steps = []
mse_values = []
anomaly_points_x = []
anomaly_points_y = []

# Model ve scaler tekrar yükle
autoencoder = load_model("lstm_autoencoder_model.keras")
scaler = joblib.load("scaler.pkl")
mean_mse = joblib.load("mean_mse.pkl")
std_mse = joblib.load("std_mse.pkl")

ser = serial.Serial(port, baud_rate, timeout=1)
print("🔵 Seri port açıldı, gerçek zamanlı veri dinleniyor...")

line_normal, = ax.plot([], [], color='blue', label='MSE')
scatter_anomaly = ax.scatter([], [], color='red', label='Anomali')

ax.set_xlabel("Zaman adımı")
ax.set_ylabel("MSE")
ax.set_title("Gerçek Zamanlı Anomali Tespiti")
ax.legend()
ax.grid(True)

t = 0

try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
        except UnicodeDecodeError:
            continue  # UTF-8 decode hatası varsa satırı atla ve devam et

        if line == "":
            continue

        try:
            value = float(line)
        except:
            continue

        data_window.append(value)

        if len(data_window) == window_size:
            scaled_window = scaler.transform(np.array(data_window).reshape(-1, 1)).reshape(1, window_size, 1)
            X_pred = autoencoder.predict(scaled_window)
            mse = np.mean(np.square(scaled_window - X_pred))
            z_score = (mse - mean_mse) / std_mse if std_mse > 0 else 0

            time_steps.append(t)
            mse_values.append(mse)
            if z_score > threshold_z:
                anomaly_points_x.append(t)
                anomaly_points_y.append(mse)

            line_normal.set_data(time_steps, mse_values)

            scatter_anomaly.remove()
            scatter_anomaly = ax.scatter(anomaly_points_x, anomaly_points_y, color='red')

            if len(time_steps) > 300:
                ax.set_xlim(time_steps[-300], time_steps[-1])
            else:
                ax.set_xlim(0, max(300, time_steps[-1]))

            all_y = mse_values + anomaly_points_y
            y_min = min(all_y) if all_y else 0
            y_max = max(all_y) if all_y else 1
            ax.set_ylim(y_min - 0.01, y_max + 0.01)

            fig.canvas.draw()
            fig.canvas.flush_events()

            t += 1

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n🔴 Seri port dinleme sonlandırıldı.")

finally:
    ser.close()
    plt.ioff()
    plt.show()





















