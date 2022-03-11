import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#Random Selection (Rasgele Seçim)
"""
import random

N = 10000
d = 10 
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
    
    
plt.hist(secilenler)
plt.show()
"""
#UCB
import math

N = 10000 #10000 tıklama
d = 10 #10 tane ilan

#Ri(n)
oduller = [0] * d #ilk basta bütün ilan ölülü 0
#Ni(n)
tiklamalar = [0] * d # o ana kadarki tıklamalar
toplam = 0 #Toplam Ödül

secilenler = []

for n in range(1,N):
    ad = 0 #Seçilen ilan
    max_ucb = 0
    for i in range(0,d):
        
        if(tiklamalar[i] > 0):
            
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt((3/2) * (math.log(n,10) / tiklamalar[i]))
            ucb = ortalama + delta
        else:
            ucb = N * 10
        
        if max_ucb < ucb: #Maxtan Büyük bir ucb çıktı
            max_ucb = ucb
            ad = i
        
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul

print("Toplam Ödül: ")
print(toplam)

plt.hist(secilenler)
plt.show()