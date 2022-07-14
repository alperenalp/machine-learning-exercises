# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:43:09 2022

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

#LogisticRegression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


