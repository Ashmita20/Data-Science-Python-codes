# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 23:35:09 2020

@author: user
"""


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

cars = pd.read_csv("E:\\Datasets\\probabililty\\wc-at.csv")
np.mean(cars.Waist)
np.std(cars.Waist)
plt.hist(x=cars.Waist,rwidth=0.9)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Waist')
plt.ylabel('Frequency')
np.mean(cars.AT)
np.std(cars.AT)
plt.hist(x=cars.AT,rwidth=0.9)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('AT')
plt.ylabel('Frequency')