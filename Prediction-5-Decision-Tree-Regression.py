# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:43:01 2022

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





