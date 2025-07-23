import numpy  as np 
import matplotlib.pyplot as plt
zaman = np.linspace(0,50,1000)#0 dan 50 ye kadar 1000 noktası
sinyal = np.sin(zaman)

gürültü = np.random.normal(0,0.1, size=zaman.shape)
sinyal_gurultu = sinyal + gürültü


plt.plot(zaman , sinyal_gurultu,label = "Gürültülü Sinyal")
plt.title("Sinyal Grafiği")
plt.xlabel("Zaman")
plt.ylabel("Sinyal Değeri")
plt.legend()
plt.grid(True)
#plt.show()
 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()   
olceklenmis_sinyal = scaler.fit_transform(sinyal_gurultu.reshape(-1,1))
print("Ölçeklenmiş Sinyal: " ,olceklenmis_sinyal[:10])

def pencere_oluştur(sinyal_gurultu , pencere_boyutu =20):
    veri_pencereleri= []
    for i in range(len(sinyal_gurultu) - pencere_boyutu ):
        veri_pencereleri.append(sinyal_gurultu[i:i+ pencere_boyutu])
    return np.array(veri_pencereleri)
pencere_boyutu = 20
veri_pencereleri = pencere_oluştur(olceklenmis_sinyal.flatten(),pencere_boyutu=20)
print(f"Veri penceereleri:{veri_pencereleri[:5]}")

from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.1,random_state=42)
model.fit(veri_pencereleri)
tahminler =model.predict (veri_pencereleri)

anomaliler = np.ones_like(sinyal_gurultu)
anomaliler[20:] =tahminler

plt.plot(zaman,sinyal_gurultu,label="Anomali Sinyal Tespiti")
plt.scatter(zaman[anomaliler==-1], sinyal_gurultu[anomaliler==-1],c="blue",label="Anomali",s=20)
plt.legend()
plt.title("Anomali Tespiti ")
plt.grid(True)
plt.show()
print(f"Toplam veri: {len(sinyal_gurultu)}")
print(f"Pencere sayısı: {len(veri_pencereleri)}")
print(f"Tahmin boyutu: {len(tahminler)}")
print(f"Anomaliler boyutu: {len(anomaliler[pencere_boyutu:])}")
