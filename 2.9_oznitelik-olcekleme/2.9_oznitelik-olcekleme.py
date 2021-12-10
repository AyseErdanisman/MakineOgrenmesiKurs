# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:27:12 2021

@author: aysee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy', 'kilo']]
print(boykilo)

class insan:
    boy = 180
    def Kosmak(self, b):
        return b + 10
    
ali = insan()
print(ali.boy)
print(ali.Kosmak(90))

"""
tabloda bulunan eksik veriler için her bosluga fix bir deger atanabilir ama biz
eksik verinin bulundugu kolonun ortalamsını alıp ortalama bir deger ataması 
yapacagiz
"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
# :, => bütün satirlarin alinmasini istediğimiz için 
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
#fit egitmek icin kullanilir - ortalama degerleri ogrendi
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
#transform ile de ogrenilen degerler nan ile degistirdi
print(Yas)

#kisaca fit ogretmek için transform ise uygulatmak icin kullanılır

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(list(range(22)))
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr', 'us'])
"""datafarme'lerin diziden en büyük farkı index ve kolon başlıklarının olmasıdır, dizileride 
kolon başlıklarından söz edemeyiz"""
# index ile her satırı 0 1 ... 22 ye kadar adlandırdık, columns ile de her sütüna isim verdik 
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns=['boy', 'kilo', 'yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
# verilerdeki cinsiyet kolonunu aldık

sonuc3 = pd.DataFrame(data = cinsiyet, index=range(22), columns=['cinsiyet'])
print(sonuc3)

# 3 tane dataframe oluşturduk: sonuc, sonuc2, sonuc3. Şimdi bunların birleştirilmesi gerek

s = pd.concat([sonuc, sonuc2], axis=1)
# axis=1 -> kolonu sağa ekler
# axis=0 -> kolonu aşağı ekler
print(s)
# alt dataframeleri alıp birleştirdik
"""normalde dikey birleştirme yapılır; birbirine denk gelen kolonlar alt alta yazılır, 
karşılığı yoksa nan değer kabul edilir ama biz axis=1 diuyerek satır başlıklarından 
eşleştirmeyi yaptık"""

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

"""oluşturulan yeni dataframein eski veriler tablosundan farkı veriler üzerinde yapılan
değişikliklerin(mesela 28.5 gibi) yeni tabloda gözükmesi
"""


#dataframei eğitim ve test kümelerine bölelim
#ülke, boy, kilo ve yaşı ayrı bir dataframe de cinsiyeti ayrı bir dataframe de toplayalım
 
from sklearn.model_selection import train_test_split
"""veriyi 4 parçaya böleceğiz
x = bağımsız değişkenleri 
y = bağımlı değişkenleri ifade edecek
tarin ve test kısmında da öğrenme ve uygulama gerçekleşecek"""

# ve eklemeler yapılan son dataframe kullanılacak

x_train, x_test, y_tarin, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)
# test_size = 33 -> %67 tarin, %33 test olarak bölündü 


"""boy kilo ve yaş sütünlarındaki değerler birbirinden farlı aralıklara sahip(mesela 
ortalamalar, standart sapmalar veya min değer ile max değer arasındaki farklar gibi değerler 
her sütün için birbirinde çok farlı), biz burada makine öğrenmesi algoritması kullanırken 
bunları daha birbirine yakın değerler dönüştürmek istersek: """

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)













