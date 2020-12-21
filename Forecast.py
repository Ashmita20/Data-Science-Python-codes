# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:43:42 2020

@author: CHARU NANDAN
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
Airlines =pd.read_excel("E:\\Datasets\\Forecast\\Airlines.xlsx")
Airlines['Month']=pd.to_datetime(Airlines['Month'],infer_datetime_format=True)
# Setting index as Month
indexedDataset = Airlines.set_index(['Month'])
from datetime import datetime
indexedDataset.head(10)
plt.xlabel("Date")
plt.ylabel("NUmber of passengers")
plt.plot(indexedDataset)
#Determining rolling statistics to check data is stationary or not
rolmean = indexedDataset.rolling(window=12).mean() # Window 12 means as 12 months for days 365
rolstd=indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)
orig=plt.plot(indexedDataset,color='blue',label='Original')
rolm=plt.plot(rolmean,color='red',label='Rolling Mean')
rols=plt.plot(rolstd,color='black',label='Rolling Standard Deviation')
plt.title("Rolling mean and Standard Deviation")
plt.legend(loc='best')# for getting the color codes and labels
plt.show(block=True)
# Dickey Fuller test
from statsmodels.tsa.stattools import adfuller
print("Results of Dicky Fuller Test")
dftest= adfuller(indexedDataset['Passengers'],autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistics','p-value','lags Used','Number of Observations'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key]=value
print(dfoutput)    
indexedDatasetlog = np.log(indexedDataset)
plt.plot(indexedDatasetlog)
movingAveg = indexedDatasetlog.rolling(window=12).mean() # Window 12 means as 12 months for days 365
movingStd =indexedDatasetlog.rolling(window=12).std()
plt.plot(indexedDatasetlog)
plt.plot(movingAveg,color='red')
plt.plot(movingStd,color='black')
datasetlogscaleMinusMovingAveg= indexedDatasetlog - movingAveg
datasetlogscaleMinusMovingAveg.head(10)
datasetlogscaleMinusMovingAveg.dropna(inplace=True)
datasetlogscaleMinusMovingAveg.head(10)
def test_stationary(timeseries):
    rolmean = timeseries.rolling(window=12).mean() # Window 12 means as 12 months for days 365
    rolstd=timeseries.rolling(window=12).std()
    print(rolmean,rolstd)
    orig=plt.plot(timeseries,color='blue',label='Original')
    rolm=plt.plot(rolmean,color='red',label='Rolling Mean')
    rols=plt.plot(rolstd,color='black',label='Rolling Standard Deviation')
    plt.title("Rolling mean and Standard Deviation")
    plt.legend(loc='best')# for getting the color codes and labels
    plt.show(block=True)
    # Dickey Fuller test
    from statsmodels.tsa.stattools import adfuller
    print("Results of Dicky Fuller Test")
    dftest= adfuller(timeseries['Passengers'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistics','p-value','lags Used','Number of Observations'])
    for key,value in dftest[4].items():
      dfoutput['Critical Value (%s)'%key]=value
      print(dfoutput)
test_stationary(datasetlogscaleMinusMovingAveg)
exponentialDecayWeightedAverage=indexedDatasetlog.ewm(halflife=12).mean()
plt.plot(indexedDatasetlog)
plt.plot(exponentialDecayWeightedAverage)
datasetlogscaleMinusexponentialDecayWeightedAverage= indexedDatasetlog - exponentialDecayWeightedAverage
test_stationary(datasetlogscaleMinusexponentialDecayWeightedAverage)
datasetlogShifting= indexedDatasetlog - indexedDatasetlog.shift()
plt.plot(datasetlogShifting)
datasetlogShifting.dropna(inplace=True)
test_stationary(datasetlogShifting)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDatasetlog)
trend=decomposition.trend
residual=decomposition.resid
seasonal= decomposition.seasonal 
plt.subplot(411)
plt.plot(indexedDatasetlog,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')
plt.tight_layout()
decompoasedlog = residual
decompoasedlog.dropna(inplace=True)
test_stationary(decompoasedlog)
# ACF plots and PACF plots on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(datasetlogShifting,lags=12,use_vlines=True)
tsa_plots.plot_pacf(datasetlogShifting,lags=12,use_vlines=True)
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(indexedDatasetlog, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(datasetlogShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetlogShifting['Passengers'])**2))
model = ARIMA(indexedDatasetlog, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(datasetlogShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetlogShifting['Passengers'])**2))
model = ARIMA(indexedDatasetlog, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(datasetlogShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetlogShifting['Passengers'])**2))
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(indexedDatasetlog['Passengers'], index=indexedDatasetlog.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
predictions_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)
# ARIMA model has taken the exact graph feature 
results_ARIMA.plot_predict(1,216)
# number of rows + (12*number of years for which you want forecast)
#where 12 is number of months in a year here 96rows and 12*10=120 so 120+96=216
results_ARIMA.forecast(120)
