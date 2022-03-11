#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

veriler = pd.read_csv('mls-salaries-2017.csv')


#Eksik Verileri Silme

veriler.dropna(inplace = True)

#Kulüpleri Numerik Yapma
kulüp = veriler.iloc[:,0:1].values

from sklearn import preprocessing

ohe = preprocessing.OneHotEncoder()
kulüp = ohe.fit_transform(kulüp).toarray()

#Pozisyonları Numerik Yapma
pozisyon = veriler.iloc[:,3:4].values

from sklearn import preprocessing

ohe = preprocessing.OneHotEncoder()
pozisyon = ohe.fit_transform(pozisyon).toarray()

#Numpy Dizileri Dataframe Dönüşümü
s = pd.DataFrame(data = kulüp, index=(range(610)), columns=["ATL","CHI","CLB","COL","DAL","DC","HOU","KC","LA","LAFC","MNUFC","MTL","NE","NYCFC","NYRB","ORL","PHI","POR","RLS","SEA","SJ","TOR","VAN"])
s2 = pd.DataFrame(data = pozisyon, index=(range(610)),columns=["D","D-M","F","F-M","F/M","GK","M","M-D","M-F","M/F"])


Base_salary = veriler.iloc[:,4:5].values
Guaranteed_compensation = veriler.iloc[:,-1:].values

Base_salary = pd.DataFrame(data = Base_salary, index=(range(610)),columns=["Base_salary"])
Guaranteed_compensation = pd.DataFrame(data = Guaranteed_compensation, index=(range(610)),columns=["Guaranteed_compensation"])

#Dataları Birleştirmek
s3 = pd.concat([s,s2], axis=1)
Sonveri = pd.concat([s3,Base_salary],axis=1)


#verilerin egitim ve test icin bolunmesi
#from sklearn.model_selection import train_test_split

#x_train, x_test,y_train,y_test = train_test_split(Sonveri,Guaranteed_compensation,test_size=0.33, random_state=0)

"""
#Model İnşaası
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()


plt.plot(x_train, y_train)
plt.plot(x_test,lr.predict(x_test))
"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(Sonveri,Guaranteed_compensation)
model = sm.OLS(lin_reg.predict(Sonveri),Sonveri)
print(model.fit().summary())
print("Linear R2 degeri:")
print(r2_score(Y, lin_reg.predict((Sonveri))))