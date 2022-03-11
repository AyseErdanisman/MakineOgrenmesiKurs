import numpy as np
import pandas as pd

yorumlar = pd.read_csv("Restaurant_Reviews.csv", error_bad_lines=False)
yorumlar.dropna(inplace = True)
yorumlar.reset_index(inplace = True)
import re

#Stop Word Kelimeleri(in, the, that) Sil
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords

#Kelimeyi Köklerine Ayır
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


#Preprocessing(Önişleme)
derlem = []
for i in range(704):
    yorum = re.sub("[^a-zA-Z]"," ",yorumlar["Review"][i]) #Boşlukları Sil
    yorum = yorum.lower() #Bütük Kelimeleri Küçük Harfle Yaz
    yorum = yorum.split() #Her bir Kelimeyi Listeye At
    
    stop_words = set(stopwords.words('english')) 
    stop_words.remove('not')
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in stop_words]
    """
    Elemanları Stopword kelimelerinden oluşan bir kümede şayet kelime yoksa 
    o zaman bu kelimeyi gövdesini bul ve bunu ilk eleman yap
    """
    
    yorum = " ".join(yorum) #Yorumu string yaptık aralarına boşluk koyarak
    derlem.append(yorum)
    
#Veriyi 0 ve 1 haline getirdik (Makine Öğrenmesine Hazır Hale Getirdik) 
#Öznitelik Çıkarımı
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray() #Bağımsız Değişken
Y = yorumlar.iloc[:,2] #Bağılı Değişken

#Sınıflandırma Algoritması Naive_Bayes
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) # %73,25 Başarı Oranı