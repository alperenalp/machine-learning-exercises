#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')

#test
print(veriler)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)

#encoder: Kategorik -> numerik
ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


# Cinsiyeti 1 ve 0'lar haline getirme
c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)
 

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns=["cinsiyet"])
print(sonuc3)


#dataframe birlestirme islemi
s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)


#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)




