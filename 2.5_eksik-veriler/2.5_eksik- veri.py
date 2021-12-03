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
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
#fit egitmek icin kullanilir - ortalama degerleri ogrendi
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
#transform ile de ogrenilen degerler nan ile degistirdi
print(Yas)













