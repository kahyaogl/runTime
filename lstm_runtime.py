# realtime_plot.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import serial
from collections import deque

# --- Parametreler ---
port = "COM4"
baud_rate = 230400
window_size = 50
threshold_z = 5

data_window = deque(maxlen=window_size)

# --- Grafik ayarları ---
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel("Zaman Adımı")
ax.set_ylabel("Z-score")
ax.set_title("Gerçek Zamanlı Anomali Tespiti (Z-score)")
ax.grid(True)
line_z, = ax.plot([], [], label='Z-score', color='blue')
scatter_anomalies = ax.scatter([], [], color='red', label='Anomali')
ax.legend()

# --- Model ve scaler yükle ---
autoencoder = load_model("lstm_autoencoder_model.keras")
scaler = joblib.load("scaler.pkl")
mean_mse = joblib.load("mean_mse.pkl")
std_mse = joblib.load("std_mse.pkl")
min_val = scaler.data_min_[0]
scale = scaler.scale_[0]

# --- Gerçek zamanlı veri akışı ---
ser = serial.Serial(port, baud_rate, timeout=1)
print("🔵 Veri akışı başlatıldı...")

# --- Değişkenler ---
z_score_values = []
anomaly_x = []
anomaly_y = []
x_values = []
t = 0

try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
        except UnicodeDecodeError:
            continue

        if not line:
            continue

        try:
            value = float(line)
        except:
            continue

        data_window.append(value)

        if len(data_window) == window_size:
            scaled = (np.array(data_window) - min_val) / scale
            seq = scaled.reshape(1, window_size, 1)
            prediction = autoencoder.predict(seq)
            mse = np.mean(np.square(seq - prediction))
            z_score = (mse - mean_mse) / std_mse if std_mse > 0 else 0

            x_values.append(t)
            z_score_values.append(z_score)

            if z_score > threshold_z:
                anomaly_x.append(t)
                anomaly_y.append(z_score)

            line_z.set_data(x_values, z_score_values)
            scatter_anomalies.remove()
            scatter_anomalies = ax.scatter(anomaly_x, anomaly_y, color='red', label="Anomali")

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






















