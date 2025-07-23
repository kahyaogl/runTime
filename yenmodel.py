# === MODEL EĞİTİMİ (Sadece bir kez çalışır) ===
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Eğitim verisini oku
df = pd.read_csv('onceseslisonrasessizsonrasesli.csv')
train_values = df.values.reshape(-1, 1)

# Normalizasyon
scaler = StandardScaler()
scaled_train = scaler.fit_transform(train_values)

# Model eğitimi
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(scaled_train)

# === SERİ PORT + GRAFİK ===
import serial
import time
import numpy as np
import matplotlib.pyplot as plt

PORT = 'COM4'
BAUDRATE = 230400
READ_INTERVAL = 0.1 # saniye

def read_serial_lines(ser, interval):
    data = []
    start_time = time.time()
    while (time.time() - start_time) < interval:
        if ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            try:
                value = float(line)
                data.append(value)
            except:
                pass  # Hatalı satırı atla
        else:
            time.sleep(0.01)
    return data

def detect_anomalies_model(data):
    if len(data) == 0:
        return np.array([])
    data = np.array(data).reshape(-1, 1)
    scaled_data = scaler.transform(data)
    preds = model.predict(scaled_data)  # 1: normal, -1: anomali
    anomalies = preds == -1
    return anomalies

def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # Port açılması
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    try:
        while True:
            data = read_serial_lines(ser, READ_INTERVAL)
            if not data:
                print("Veri gelmedi, tekrar dene...")
                continue
            
            anomalies = detect_anomalies_model(data)
            x = np.arange(len(data))
            
            ax.clear()
            ax.plot(x, data, 'b-', label='Veri')
            if anomalies.any():
                ax.scatter(x[anomalies], np.array(data)[anomalies], color='red', label='Anomali')
            ax.set_title(f'Anomali Tespiti ({READ_INTERVAL})')
            ax.set_xlabel('Data Noktası')
            ax.set_ylabel('Değer')
            ax.legend()
            plt.pause(0.01)
            
            print(f"Son {(READ_INTERVAL) }saniye veri sayısı: {len(data)}, Anomali sayısı: {np.sum(anomalies)}")
    
    except KeyboardInterrupt:
        print("Program durduruldu.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
