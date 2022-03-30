# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:04:40 2022

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


# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR
# diger bir tahmin cesidi rbf'e göre tahmin ettirme. istersek "RBF", "Linear", "Polynomial" da seçilebilir.
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)
svr_tahmin = svr_reg.predict(x_olcekli)

#Görselleştirme
plt.scatter(x_olcekli, y_olcekli, color="blue")
plt.plot(x_olcekli, svr_tahmin, color="red")
plt.show()

print("\ Support Vector Regresyon 11 ve 6 indexine göre tahmin sonuçları")
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))








