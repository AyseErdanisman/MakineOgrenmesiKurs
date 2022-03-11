#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')

hava = veriler.iloc[:,1:3]


#encoder: Kategorik -> Numeric

from sklearn import preprocessing

outlook = veriler.iloc[:,0:1].values
le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(veriler.iloc[:,0])


ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

#encoder: Kategorik -> Numeric

windy = veriler.iloc[:,-2:-1].values
le = preprocessing.LabelEncoder()

windy = le.fit_transform(windy)


#encoder: Kategorik -> Numeric

play = veriler.iloc[:,-1:].values
le = preprocessing.LabelEncoder()

play[:,0] = le.fit_transform(play[:,0])


#numpy dizileri dataframe donusumu
havadurumu = pd.DataFrame(data=outlook, index = range(14), columns = ['overcast','rainy','sunny'])


sicaklik = pd.DataFrame(data=hava, index = range(14), columns = ['temperature','humidity'])

ruzgar = pd.DataFrame(data = windy, index = range(14), columns = ['windy'])

oynanabilir = pd.DataFrame(data = play, index = range(14), columns = ["play"])



#dataframe birlestirme islemi
s=pd.concat([havadurumu,sicaklik], axis=1)

s2=pd.concat([s,ruzgar], axis=1)

sonveri = pd.concat([s2,oynanabilir], axis = 1)



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s2,play,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)



#Geri Eleme
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = sonveri, axis=1)

X_l = sonveri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)

model = sm.OLS(sonveri.iloc[:,-1].values,X_l).fit()
print(model.summary())
"""
import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test
"""

#Hocanın Yaptığı Yol
"""
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)
"""