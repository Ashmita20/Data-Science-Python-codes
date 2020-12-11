# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 23:09:51 2020

@author: user
"""


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt 
Q9 = pd.read_csv("E:\\Datasets\\Mean median mode\\Q9_a.csv")
Q9.skew(axis=0 , skipna= True)
Q9.kurtosis(axis=0,skipna=True)
Q9b = pd.read_csv("E:\\Datasets\\Mean median mode\\Q9_b.csv")
Q9b.skew(axis=0 , skipna= True)
Q9b.kurtosis(axis=0,skipna=True)