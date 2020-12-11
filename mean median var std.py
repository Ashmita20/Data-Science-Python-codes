# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:56:41 2020

@author: Ashmita
"""


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt 
Q7 = pd.read_csv("E:\\Datasets\\Mean median mode\\Q7.csv")
data=[]
np.mean(Q7.Points)
np.median(Q7.Points)
np.var(Q7.Points)
np.std(Q7.Points)

np.mean(Q7.Score)
np.median(Q7.Score)
np.var(Q7.Score)
np.std(Q7.Score)

np.mean(Q7.Weigh)
np.median(Q7.Weigh)
np.var(Q7.Weigh)
np.std(Q7.Weigh)

data = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
np.mean(data)
print(np.std(data))


