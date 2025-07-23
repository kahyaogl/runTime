""""from sklearn.preprocessing import StandardScaler
import numpy as np 
#orijinal_veri = np.array([[10,100], [20,200],[30,300]])#2 özellik

#scaler = StandardScaler()
#ölçeklenmis_veri =scaler.fit_transform(orijinal_veri)
#print ("ölçeklenmis_veri : \n"  , ölçeklenmis_veri)

import pandas as pd

veri = {
    'ad': ['Ali', 'Veli', 'Ayşe'],
    'yaş': [23, 34, 29],
    'puan': [85, 90, 95]
}

df = pd.DataFrame(veri)
print(df)
# DataFrame oluşturma(excel tablosu gibi)
from sklearn.model_selection import train_test_split
veri_ozellikleri = [[1] ,[2],[3],[4], [5],[6]]
hedef_sınıflar = [0,0,0,1,1,1]

X_egitim,X_test,y_egitim,y_test=train_test_split(veri_ozellikleri,hedef_sınıflar,test_size=0.33,random_state=42)
print("Eğitim Özellikleri:", X_egitim)
print("Eğitim Hedefleri:", y_egitim)
print("Test Özellikleri:", X_test)
print("Test Hedefleri:", y_test)

from sklearn.linear_model import LogisticRegression

X_egitim = [[0],[1],[2],[3]]
y_egitim= [0,0,1,1]
model = LogisticRegression()
model.fit(X_egitim,y_egitim)
yeni_veri =[[1.5] ,[2.5]]
tahminler = model.predict(yeni_veri)
print("yeni veriler için tahminler :",tahminler)

from sklearn.ensemble import RandomForestClassifier

veri_ozellikleri = [[0,0],[1,1],[0,1],[1,0]]
hedef_siniflar = [0,1,1,0]

rastgele_orman = RandomForestClassifier(n_estimators=10,random_state=42)
rastgele_orman.fit(veri_ozellikleri, hedef_siniflar)

yeni_ornek = [[0.8,0.8]]
tahmin = rastgele_orman.predict(yeni_ornek)

print("tahmin :",tahmin)  

from sklearn.svm import OneClassSVM
veri_ozellikleri = [[0,0] ,[1,1], [1,0],[0,1]]
hedef_siniflar = [0 ,1,1,0]
svm_modeli = OneClassSVM(kernel='linear', nu=0.1)
svm_modeli.fit(veri_ozellikleri,hedef_siniflar)
yeni_veri =[[0.9,0.6]]
tahmin =svm_modeli.predict([yeni_veri])
print("SVM tahmini.",tahmin)

from sklearn.tree import DecisionTreeClassifier
veri_ozellikleri = [[160,50],[170,65],[180,85]]
hedef_siniflar = ['zayıf','normal','kilolu']

model = DecisionTreeClassifier()
model.fit(veri_ozellikleri,hedef_siniflar)
yeni_ornek = [[175,70]]

tahmin = model.predict(yeni_ornek)
print("Decision Treee tahmini : " ,tahmin)

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
veri  = np.array([[170, 70], [160, 60], [180, 80], [155, 50], [165, 55]])



kmeans =KMeans(n_clusters=3,random_state=42)
kume_etiketleme = kmeans.fit_predict(veri)
merkezler = kmeans.cluster_centers_
plt.figure(figsize=(8,6))
plt.scatter(veri[:,0] , veri[:,1], c=kume_etiketleme, cmap='viridis',s=100)
plt.scatter(merkezler[:,0], merkezler[:,1],c='red',marker='X',s=200,label = 'Merkezler')
plt.xlabel("Boy (cm)")
plt.ylabel("Kilo (kg)")
plt.title("Kmeans Kümeleme")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [2, 4, 1, 3]

plt.plot(x, y, color='blue', marker='o', linestyle='--')
plt.title("Çizgi Grafiği")
plt.xlabel("Gün")
plt.ylabel("Sıcaklık")
plt.grid(True)
plt.show()"""
import matplotlib.pyplot as plt
x = [170, 165, 180, 175]
y = [70, 60, 85, 75]

plt.scatter(x, y, color='green', s=100)
plt.title("Boy vs Kilo")
plt.xlabel("Boy (cm)")
plt.ylabel("Kilo (kg)")
plt.grid(True)
plt.show()


