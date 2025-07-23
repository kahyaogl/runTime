import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

port = 'COM4'
baudrate = 230400
ser = serial.Serial(port, baudrate, timeout=1)

window_size = 50
data = deque([0]*window_size, maxlen=window_size)
anomaly_points = []

fig, ax = plt.subplots()
x = np.arange(window_size)  # x ekseni 0..49 sabit
line, = ax.plot(x, data, label='Değer')
scatter = ax.scatter([], [], color='red', label='Anomali')
ax.set_ylim(-10000, 10000)
ax.legend()

def update(frame):
    if ser.in_waiting > 0:
        try:
            raw_line = ser.readline()
            decoded_line = raw_line.decode('utf-8', errors='ignore').strip()
            filtered = ''.join(c for c in decoded_line if c.isdigit() or c in ['-', '.'])
            if filtered in ['', '-', '.']:
                return line, scatter

            value = float(filtered)
            data.append(value)

            arr = np.array(data)
            mean = arr.mean()
            std = arr.std()
            threshold = 3 * std

            # Anomali kontrolü
            # Window içindeki son değer indexi sabit: window_size-1
            if abs(value - mean) > threshold:
                # Her frame güncellendiği için duplicate olabilir, temizle
                # Burada sadece son değeri kontrol edip ekliyoruz
                anomaly_points.append(window_size - 1)
            else:
                # Anomali değilse, son indeksi kaldır
                if anomaly_points and anomaly_points[-1] == window_size - 1:
                    anomaly_points.pop()

            # Sabit x ekseni, y verisi güncelle
            line.set_ydata(data)

            ymin = min(data)
            ymax = max(data)
            if ymin == ymax:
                ymax = ymin + 1
            ax.set_ylim(ymin - abs(ymin)*0.1, ymax + abs(ymax)*0.1)

            # Anomalilerin koordinatlarını hazırla
            xs = anomaly_points
            ys = [data[i] for i in anomaly_points] if anomaly_points else []

            scatter.set_offsets(np.c_[xs, ys] if xs else [])

        except Exception as e:
            print(f"Hata: {e}")

    return line, scatter

ani = animation.FuncAnimation(fig, update, interval=50)
plt.show()



