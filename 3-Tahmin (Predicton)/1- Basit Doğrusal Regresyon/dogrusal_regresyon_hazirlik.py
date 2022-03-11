#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')

#veri on isleme

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#Model İnşası (Linear regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

#İndexleri Sıraladık Tablo Düzgün Çıksın Diye
x_train = x_train.sort_index()
y_train = y_train.sort_index()


plt.plot(x_train, y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")