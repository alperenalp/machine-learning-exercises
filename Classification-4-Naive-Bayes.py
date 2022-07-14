# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:40:03 2022

@author: Alperen
"""


# kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')
print(veriler)

x = veriler.iloc[:, 1:4].values #bagimsiz degiskenler
y = veriler.iloc[:,4:].values #bagimli degiskenler #hedef

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#NaiveBayes
#GaussianNB tahmin edilecek değerler reel sayılar olabiliyorsa ama reel sayılar ikilik sayılara indirgenebilir
#MultinomialNB isimlendirme yapılarak sınıflandırılıyorsa A,B,C,D,F gibi elma armut ayva gibi
#BernoulliNB binary değerler ise 0 yada 1 gibi değerler alıyorsa erkek kadın gibi
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 