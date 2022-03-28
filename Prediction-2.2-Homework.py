#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#veriseti yükleme
veriler = pd.read_csv('odev_tenis.csv')



#test
print(veriler)




#encoder: Kategorik -> numerik
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

#tüm verileri encod cevirme
encodVeriler = veriler.apply(preprocessing.LabelEncoder().fit_transform)


#hava durumunu sayısal hale getirme
hava = veriler.iloc[:,0:1].values
print(hava)

hava[:,0] = le.fit_transform(veriler.iloc[:,0])
print(hava)

ohe = preprocessing.OneHotEncoder()
hava = ohe.fit_transform(hava).toarray()
print(hava)


# rüzgar durumunu sayısal hale getirme
windy = veriler.iloc[:,-2].values
print(windy)

windy = le.fit_transform(veriler.iloc[:,-2])
print(windy)


# oynanırlık durumunu sayısal hale getirme
play = veriler.iloc[:,-1:].values
print(play)

play[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(play)

 



#numpy dizileri dataframe donusumu
havaDurumu = pd.DataFrame(data = hava, index = range(14), columns=["overcast","rainy","sunny"])
print(havaDurumu)

ruzgarDurumu = pd.DataFrame(data = windy, index = range(14), columns=["windy"])
print(ruzgarDurumu)

oynanirlikDurumu = pd.DataFrame(data = play, index = range(14), columns=["play"])
print(oynanirlikDurumu)





#dataframe birlestirme islemi
b1= pd.concat([havaDurumu, veriler.iloc[:,1:3]], axis=1)
print(b1)

'''
b2 = pd.concat([b1, ruzgarDurumu], axis=1)
print(b2)

sayisalVeriler= pd.concat([b2, oynanirlikDurumu], axis=1)
print(sayisalVeriler)
'''

sayisalVeriler = pd.concat([b1, encodVeriler.iloc[:,-2:]], axis=1)
print(sayisalVeriler)





# tahmini yapılması istenen değişkeni ayırma
from sklearn.model_selection import train_test_split

nem = sayisalVeriler.iloc[:,4:5].values
print(nem)
sol = sayisalVeriler.iloc[:,:4]
sag = sayisalVeriler.iloc[:,5:]

sonVeri = pd.concat([sol,sag],axis = 1)


#verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(sonVeri, nem, test_size=0.33, random_state=0)


# eğitim işlemi
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# tahmin işlemi
y_pred = regressor.predict(x_test)


# backward elemination
import statsmodels.api as sm
X = np.append(arr = np.ones((14, 1)).astype(int), values = sonVeri, axis = 1)

X_l = sonVeri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(nem, X_l).fit()
print(model.summary())


# elenen değişkeni veri setinden çıkarma
sonVeri1 = sonVeri.iloc[:,:4]
sonVeri2 = sonVeri.iloc[:,5:]
sonVeri = pd.concat([sonVeri1, sonVeri2], axis=1)

import statsmodels.api as sm
X = np.append(arr = np.ones((14, 1)).astype(int), values = sonVeri, axis = 1)

X_l = sonVeri.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(nem, X_l).fit()
print(model.summary())


#verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(sonVeri, nem, test_size=0.33, random_state=0)

# eğitim işlemi
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# tahmin işlemi
y_pred = regressor.predict(x_test)