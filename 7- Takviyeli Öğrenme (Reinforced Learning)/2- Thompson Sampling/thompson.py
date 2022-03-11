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
#Thompson Sampling
import random

N = 10000 #10000 tıklama
d = 10 #10 tane ilan


toplam = 0 #Toplam Ödül

secilenler = []

birler = [0] * d
sifirlar = [0] * d

for n in range(1,N):
    ad = 0 #Seçilen ilan
    max_th = 0
    for i in range(0,d):
        
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        if(rasbeta > max_th):
            max_th = rasbeta
            ad = i
        
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    
    if(odul == 1):
        birler[ad] = birler[ad] + 1
    else:
        sifirlar[ad] = sifirlar[ad] + 1
        
    toplam = toplam + odul

print("Toplam Ödül: ")
print(toplam)

plt.hist(secilenler)
plt.show()


