import pandas as pd

url = "https://bilkav.com/satislar.csv"

veriler = pd.read_csv(url)
veriler = veriler.values

X = veriler[:,0:1]
Y = veriler[:,1]

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

import pickle

dosya = "model.kayıt"

pickle.dump(lr,open(dosya,"wb"))

yüklenen = pickle.load(open(dosya,"rb"))
print(yüklenen.predict(x_test))