#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
#numpy ve pandas kütüphanelerini veriyi işlemek ve hafızada yönlendirmek için kullanıyoruz (data frame gibi sınıflar için)

# veri yukleme
veriler = pd.read_csv('pokemon.csv')
x1 = veriler.iloc[:,5:11]
x2 = veriler.iloc[:,19:22]
x = pd.concat([x1,x2], axis=1)
y = veriler.iloc[:,11:12]
X = x.values
Y = y.values

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)

y_pred = lin_reg.predict(x_test)

model = sm.OLS(y_pred,y_test)
print(model.fit().summary())
print(r2_score(y_test, y_pred))
