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

# linear regression 
# dogrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X, Y)
tahmin_lin = lin_reg.predict(X)

#Görselleştirme
plt.scatter(X, Y, color="blue")
plt.plot(X, tahmin_lin, color="red")
plt.show()

# polynomial regression 
# dogrusal olmayan (nonlinear) model oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

# lineer regresyon kullanarak fit yapma eğitme
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# predict'in içine dogrusal sayi yazamayız polinomsal değerler vermek zorundayız
tahmin_poly = lin_reg2.predict(poly_reg.fit_transform(X))

#Görselleştirme
plt.scatter(X, Y, color="blue")
plt.plot(X, tahmin_poly, color="red")
plt.show()

#tahmin denemeleri
print("Doğrusal Regresyon 11 ve 6 indexine göre tahmin sonuçları")
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print("\nPolinomal Regresyon 11 ve 6 indexine göre tahmin sonuçları")
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))


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








