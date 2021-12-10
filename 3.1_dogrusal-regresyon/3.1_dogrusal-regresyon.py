# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:40:01 2021

@author: aysee
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

"""
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""

#model insaası(liner regresyon)
from sklearn.linear_model import LinearRegression
"""herhangi bir obje hakkında daha fazla bilgi almak için üzerine tıkla ctrl+ı kombinasyonu 
ile help ekranı gelecektir"""
lr = LinearRegression()
lr.fit(x_train, y_train)
#x_train den y_tarin i öğrenmesini istedik

tahmin = lr.predict(x_test)
#burada X_testten öğrenip Y_testin karşılığını bulmaya çalışacak

#y_test gerçek değerler tahmin değişkeni ise gerçekleşmesi beklenen tahmini değerler
 
#veri görselleştirme
x_train = x_train.sort_index()
#burada x_trainleri sıraldık çünkü random_state den dolayı aylar karışık sıralanmıştı
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
#x_testteki her bir değer için x_testin liner regresyondaki karşılığını(tahmin değerini) al
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
