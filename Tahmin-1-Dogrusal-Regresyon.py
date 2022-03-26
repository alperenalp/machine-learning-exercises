import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('satislar.csv')

print(veriler)

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

'''
# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

'''

#model inşası(linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) 
 
tahmin= lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar") 
plt.ylabel("Satışlar") 

