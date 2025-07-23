import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

PORT = 'COM4'
BAUDRATE = 230400
READ_INTERVAL = 3  # saniye
Z_THRESHOLD = 3.4

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
                # Satır float değilse atla
                pass
        else:
            time.sleep(0.01)  # CPU yükünü azaltmak için küçük uyku
    return data

def detect_anomalies_zscore(data, threshold=Z_THRESHOLD):
    if len(data) == 0:
        return np.array([])  # Boşsa boş döndür
    z_scores = zscore(data)
    anomalies = np.abs(z_scores) > threshold
    return anomalies

def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # Port açılması için bekle
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,5))
    
    try:
        while True:
            data = read_serial_lines(ser, READ_INTERVAL)
            if not data:
                print("Veri gelmedi, tekrar dene...")
                continue
            
            anomalies = detect_anomalies_zscore(data)
            x = np.arange(len(data))
            
            ax.clear()
            ax.plot(x, data, 'b-', label='Veri')
            if anomalies.any():
                ax.scatter(x[anomalies], np.array(data)[anomalies], color='red', label='Anomali')
            ax.set_title('3 Saniyelik Seri Port Verisi ve Anomali Tespiti (Z-score)')
            ax.set_xlabel('Örnek No')
            ax.set_ylabel('Değer')
            ax.legend()
            plt.pause(0.01)
            
            print(f"Son 3 saniye veri sayısı: {len(data)}, Anomali sayısı: {np.sum(anomalies)}")
    
    except KeyboardInterrupt:
        print("Program durduruldu.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()