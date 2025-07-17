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
window_size =50
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
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(window_size)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
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
port = "COM4"  # Seri portunu kendi cihazına göre değiştir
baud_rate = 230400
threshold_z = 5

data_window = deque(maxlen=window_size)

# Grafik setup
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel("Zaman Adımı")
ax.set_ylabel("Z-score")
ax.set_title("Gerçek Zamanlı Anomali Tespiti (Z-score)")
ax.grid(True)
line_z, = ax.plot([], [], label='Z-score', color='blue')
scatter_anomalies = ax.scatter([], [], color='red', label='Anomali')
ax.legend()

# Model ve scaler tekrar yükle
autoencoder = load_model("lstm_autoencoder_model.keras")
scaler = joblib.load("scaler.pkl")
mean_mse = joblib.load("mean_mse.pkl")
std_mse = joblib.load("std_mse.pkl")
# MinMaxScaler parametrelerini manuel dönüşüm için al
min_val = scaler.data_min_[0]
scale = scaler.scale_[0]

ser = serial.Serial(port, baud_rate, timeout=1)
print("🔵 Veri akışı başlatıldı...")

# Zaman serisi verileri
z_score_values = []
anomaly_x = []
anomaly_y = []
t = 0


try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
        except UnicodeDecodeError:
            continue

        if line == "":
            continue
        try:
            value = float(line)
        except:
            continue

        data_window.append(value)

        if len(data_window) == window_size:
            scaled_value = (np.array(data_window) - min_val) / scale
            scaled_seq = scaled_value.reshape(1, window_size, 1)

            prediction = autoencoder.predict(scaled_seq)
            mse = np.mean(np.square(scaled_seq - prediction))
            z_score = (mse - mean_mse) / std_mse if std_mse > 0 else 0
            # Test için bazı Z-score'ları negatife zorla (grafiği kontrol etmek için)
            

            z_score_values.append(z_score)

            if z_score > threshold_z:
                anomaly_x.append(t)
                anomaly_y.append(z_score)
           

            # Grafik güncelle
            line_z.set_data(range(t - len(z_score_values) + 1, t + 1), z_score_values)
            scatter_anomalies.remove()
            scatter_anomalies = ax.scatter(anomaly_x, anomaly_y, color='red', label="Anomali")

            # Anomali sayısını yaz
            [child.remove() for child in ax.get_children() if isinstance(child, plt.Text) and child.get_text().startswith("Anomali Sayısı:")]
            ax.text(0.01, 0.95, f"Anomali Sayısı: {len(anomaly_x)}", transform=ax.transAxes,
                    fontsize=12, color='red', verticalalignment='top')

            ax.set_xlim(max(0, t - 300), t + 10)

            all_y = z_score_values + anomaly_y
            if all_y:
                y_min = min(all_y) - 0.5
                y_max = max(all_y) + 0.5
                limit = max(abs(y_min), abs(y_max))
                y_min = -limit
                y_max = limit
            else:
                y_min = -1.0
                y_max = 1.0

            ax.set_ylim(y_min, y_max)
            fig.canvas.draw()
            fig.canvas.flush_events()

            t += 1

except KeyboardInterrupt:
    print("\n🔴 Seri port dinleme sonlandırıldı.")

finally:
    ser.close()
    plt.ioff()
    plt.show()

