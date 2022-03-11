#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Veriyi Oku
veriler = pd.read_csv('sepet.csv', header=None)

#Veriyi Liste İçinde Liste Haline Getir
t = []
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])

#Nan Değerlerini Temizle
cleaned_list = []
for x in t:
    clist = []
    for z in x:
        if str(z) != 'nan':
            clist.append(z)
    cleaned_list.append(clist)
    
 
#Onu Alan Bunu da Aldı Demek İçin Bir Değer Aralığı ver ve Kural Oluştur
from apyori import apriori
rules = apriori(cleaned_list, min_support = 0.01, min_confidance = 0.2, min_lift = 3)
rules_list = list(rules)

#Ekrana Daha Güzel Bastır
for item in rules_list:
 
    base_items = [x for x in item[2][0][0]]
    add_item, = item[2][0][1]
    print("Rule: " + " + ".join(base_items) + " -> " + str(add_item))
 
    print("Support: " + str(item[1]))
 
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")