# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Veri Yükleme
veriler = pd.read_csv("maaslar.csv")

#dataframe dilimleme(slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#Linear Regresyon
#doğrusal (linear) oluşturma
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)


#Polinomal Regresyon
#doğrusal olmayan (nonlinear model) oluşturma
#2. Dereceden polinom
from sklearn.preprocessing import PolynomialFeatures

poly_reg2 = PolynomialFeatures(degree = 2)
x_poly2 = poly_reg2.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2,y)


#4. Dereceden Polinom
from sklearn.preprocessing import PolynomialFeatures

poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(x)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)



#16. Dereceden Polinom
from sklearn.preprocessing import PolynomialFeatures

poly_reg4 = PolynomialFeatures(degree = 16)

x_poly4 = poly_reg4.fit_transform(x)

lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4,y)


#64. Dereceden Polinom
from sklearn.preprocessing import PolynomialFeatures

poly_reg5 = PolynomialFeatures(degree = 64)

x_poly5 = poly_reg5.fit_transform(x)

lin_reg5 = LinearRegression()
lin_reg5.fit(x_poly5,y)




#veri görselleştirme
#doğrusal linear
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color = "blue")
plt.show()

#2. Derece
plt.scatter(x,y,color = "red")
plt.plot(x,lin_reg2.predict(poly_reg2.fit_transform(x)),color = "yellow")
plt.show()

#4. Derece
plt.scatter(x,y,color = "red")
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(x)),color = "green")
plt.show()

#16. Derece
plt.scatter(x,y,color = "red")
plt.plot(x,lin_reg4.predict(poly_reg4.fit_transform(x)),color = "orange")
plt.show()

#64. Derece
plt.scatter(x,y,color = "red")
plt.plot(x,lin_reg5.predict(poly_reg5.fit_transform(x)),color = "green")
plt.show()


#Tahminler
#print(lin_reg.predict([[11]]))
#print(lin_reg.predict([[6.6]]))