# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:48:54 2022

@author: Alperen
"""


#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veriseti yükleme
veriler = pd.read_csv('maaslar.csv')

#data frame dilimleme(slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# data frame'i numpy dizisine çevirme
X = x.values
Y = y.values

# Decision Tree (Karar Agaci) regresyon kullanarak eğitme
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)
dt_tahmin = dt_reg.predict(X)
Z = X + 0.5
K = X - 0.4

#Görselleştirme
plt.scatter(X, Y, color="blue")
plt.plot(X, dt_tahmin, color="red")
plt.plot(X, dt_reg.predict(Z), color="green")
plt.plot(X, dt_reg.predict(K), color="yellow")
plt.show()

print("\nDecision Tree Regresyon 11 ve 6 indexine göre tahmin sonuçları")
print(dt_reg.predict([[11]]))
print(dt_reg.predict([[6.6]]))


# Algoritma Degerlendirme skor hesaplama
from sklearn.metrics import r2_score
print('\nDecision Tree Regresyon R2 Degeri')
print(r2_score(Y, dt_reg.predict(X)))
print(r2_score(Y, dt_reg.predict(Z)))
print(r2_score(Y, dt_reg.predict(K)))


#Random Forest(Rastgele Orman) Regresyon algoritması kullanarak eğitme
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())

#Görselleştirme
plt.scatter(X,Y)
plt.plot(X, rf_reg.predict(X),color="red")
plt.plot(X, rf_reg.predict(Z), color="green")
plt.plot(X, rf_reg.predict(K), color="yellow")
plt.show()

print("\nRandom Forest Regresyon algoritması ile 11 ve 6 indexine göre tahmin sonuçları")
print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))


# Algoritma Degerlendirme skor hesaplama
from sklearn.metrics import r2_score
print('\nRandom Forest R2 Degeri')
print(r2_score(Y, rf_reg.predict(X)))
print(r2_score(Y, rf_reg.predict(Z)))
print(r2_score(Y, rf_reg.predict(K)))












print("\nTahmin Başarılı bir şekilde yapıldı")





