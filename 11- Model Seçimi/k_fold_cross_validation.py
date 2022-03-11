#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('Social_Network_Ads.csv')

X = veriler.iloc[:,[2,3]].values
Y = veriler.iloc[:,4].values

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


#K-Katlamalı Çapraz Doğrulama
from sklearn.model_selection import cross_val_score

"""

1. Estimator: classifies(Bizim Durum İçin)
2. X
3. Y
4. CV: kaç katlamalı

"""

cvs = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=4)#Başarı

print(cvs.mean())
print(cvs.std())