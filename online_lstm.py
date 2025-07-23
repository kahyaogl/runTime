import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense, RepeatVector, TimeDistributed

# === Parametreler ===
PORT = 'COM4'
BAUDRATE = 230400
WINDOW_SECONDS = 2
INIT_EPOCHS = 2
UPDATE_EPOCHS = 0    # Güncelleme eğitimi kapalı
BATCH_SIZE = 16
ANOMALY_THRESHOLD = 0.01  # Bunu eğitim sonrası otomatik hesaplamak daha iyi
PLOT_UPDATE_INTERVAL = 2  # Grafik güncelleme sıklığı

# === Veri okuma ===
def get_sampling_rate(ser):
    count = 0
    start = time.time()
    while time.time() - start < 1:
        try:
            line = ser.readline().decode().strip()
            float(line)
            count += 1
        except:
            continue
    return count

# === Model oluştur ===
def build_model(timesteps, features):
    model = Sequential([
        GRU(32, activation='relu', input_shape=(timesteps, features), return_sequences=False),
        RepeatVector(timesteps),
        GRU(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === Model eğit ===
def train_model(model, data, epochs):
    model.fit(data, data, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
    return model

# === Rekonstrüksiyon hatası hesapla ===
def get_mse(model, data):
    recon = model.predict(data, verbose=0)
    return np.mean(np.square(data - recon))

# === Grafik ayarı ===
def init_plot(window_size):
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Veri")
    dots, = ax.plot([], [], 'ro', label="Anomali")
    ax.set_xlim(0, window_size)
    ax.set_ylim(-1, 1)
    ax.legend()
    return fig, ax, line, dots

# === Ana Döngü ===
def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)

    print("Sampling rate ölçülüyor...")
    sampling_rate = get_sampling_rate(ser)
    window_size = WINDOW_SECONDS * sampling_rate
    print(f"Sampling rate: {sampling_rate} → pencere: {window_size}")

    fig, ax, line, dots = init_plot(window_size)
    buffer = deque(maxlen=window_size)

    # İlk pencereyi oku ve modeli eğit
    while len(buffer) < window_size:
        try:
            val = float(ser.readline().decode().strip())
            buffer.append(val)
        except:
            continue

    train_seq = np.array(buffer).reshape((1, window_size, 1))
    model = build_model(window_size, 1)
    print("İlk model eğitiliyor...")
    model = train_model(model, train_seq, INIT_EPOCHS)
    print("İlk model hazır.")

    plot_update_counter = 0

    # Sürekli döngü
    while True:
        new_buffer = []
        while len(new_buffer) < window_size:
            try:
                val = float(ser.readline().decode().strip())
                new_buffer.append(val)
            except:
                continue

        test_seq = np.array(new_buffer).reshape((1, window_size, 1))

        # Anomali kontrolü
        mse = get_mse(model, test_seq)
        is_anomaly = mse > ANOMALY_THRESHOLD
        print(f"[{time.strftime('%H:%M:%S')}] MSE: {mse:.5f} → {'Anomali' if is_anomaly else 'Normal'}")

        # Grafik güncelleme (her 5 pencere bir güncelle)
        plot_update_counter += 1
        if plot_update_counter >= PLOT_UPDATE_INTERVAL:
            line.set_data(range(window_size), new_buffer)
            if is_anomaly:
                dots.set_data(range(window_size), new_buffer)
            else:
                dots.set_data([], [])
            ax.set_ylim(min(new_buffer)-0.1, max(new_buffer)+0.1)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            plot_update_counter = 0

        # Log kaydı
        with open("online_anomaly_log.csv", "a") as f:
            f.write(f"{time.time()},{mse:.6f},{int(is_anomaly)}\n")

        # Güncelleme için eğitim (kapalı)
        # model = train_model(model, test_seq, UPDATE_EPOCHS)

main()
