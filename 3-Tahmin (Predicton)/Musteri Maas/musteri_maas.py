#Aşağıda, Python ile şimdiye kadar yazdığımız tahmin algoritmalarının şablonunu bulabilirsiniz: 

#Kütüphaneler

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
#numpy ve pandas kütüphanelerini veriyi işlemek ve hafızada yönlendirmek için kullanıyoruz (data frame gibi sınıflar için)

#Veri Yükleme

# veri yukleme
veriler = pd.read_csv('musteriler.csv')
sayilar = veriler.iloc[:,2:4]


#encoder: Kategorik -> Numeric
cinsiyet = veriler.iloc[:,1:2].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
cinsiyet[:,0] = le.fit_transform(veriler.iloc[:,1:2])
cinsiyet = pd.DataFrame(data = cinsiyet, index = range(200), columns = ['cinsiyet'])


x=pd.concat([cinsiyet,sayilar], axis=1)
y = veriler.iloc[:,4:]

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#Doğrusal Regresyon
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

y_pred = lin_reg.predict(X_test)
model = sm.OLS(y_pred,X_test)

print(model.fit().summary())


#Rassal Ağaç
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)

rf_reg.fit(X_train,y_train)
y_predr = rf_reg.predict(X_test)

print('dt ols')
model5 = sm.OLS(y_predr,X_test)

print(model5.fit().summary())










