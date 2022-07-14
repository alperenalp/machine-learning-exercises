# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 19:40:03 2022

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

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy') # n_estimators parametresi kaç tane karar ağacı oluşturulacağının miktarıdır fazlası overfitting yani ezberlemeye yol açar default 10'dur
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 

# Random Forest'ın ROC Eğrisine bakalım
import sklearn.metrics as metrics
probs = rfc.predict_proba(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, probs[:,0], pos_label='e')
roc_auc = metrics.auc(fpr, tpr)
#grafiği çizme
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')
plt.show()