#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Veri Kümesi
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:,0:13].values
Y = veriler.iloc[:,13].values


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)


#PCA Dönüşümünden Gelen Locistik Regresyon
from sklearn.linear_model import LogisticRegression

#PCA Dönüşümünden Gelen LR
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#PCA Dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train2,y_train)

y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)


from sklearn.metrics import confusion_matrix

#PCA Olmadan Ortaya Çıkan Sonuç
print("gerçek ve pca'siz")
cm = confusion_matrix(y_test,y_pred)
print(cm)

#PCA Varken
print("gerçek ve pca ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm)

#PCA Sonrası / PCA Öncesi Karşılaştırma
print("pca siz ve pca li")
cm = confusion_matrix(y_test,y_pred2)
print(cm)