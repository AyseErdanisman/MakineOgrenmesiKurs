# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:27:12 2021

@author: aysee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('veriler.csv')
print(veriler)

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy', 'kilo']]
print(boykilo)