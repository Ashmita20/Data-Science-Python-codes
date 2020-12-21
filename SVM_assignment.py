# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:51:13 2020

@author: CHARU NANDAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("E://Datasets//SVM//forestfires.csv")
dataset.head()
dataset.describe()

from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
dataset['month']=le.fit_transform(dataset['month'])
dataset['day']=le.fit_transform(dataset['day'])
dataset['size_category']=le.fit_transform(dataset['size_category'])
sns.pairplot(dataset,kind="scatter")
sns.boxplot(x="month", y="size_category", data=dataset)
sns.pairplot(x="size_category", y="month", data=dataset)
sns.pairplot(dataset, 
             vars = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind',
       'rain', 'area'], 
             hue = 'size_category', diag_kind = 'kde', 
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'});
# Title 
plt.suptitle('Pair Plot of Size Category Data for Forest Fire', 
             size = 28);
print(dataset.columns)
X= dataset[['month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind',
       'rain', 'area', 'dayfri', 'daymon', 'daysat', 'daysun', 'daythu',
       'daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 'monthfeb',
       'monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov',
       'monthoct', 'monthsep']]
Y = dataset[['size_category']]
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y, test_size= 0.2)
from sklearn.svm import SVC
model = SVC(kernel="linear")
svm= model.fit(Xtrain,np.ravel(Ytrain))
y_pred=model.predict(Xtest)
cm = confusion_matrix(Ytest, y_pred,labels=[0,1])
print("Confusion Matrix")
print(cm)
result=classification_report(Ytest, y_pred)
print(result)
rheat = classification_report(Ytest, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Ytest,y_pred)
print("Accuracy:",result2)
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
gridmodel=GridSearchCV(SVC(), param_grid,refit=True,verbose=3)
gridmodel.fit(Xtrain,Ytrain)
gridmodel.best_params_
gridmodel.best_estimator_
grid_predictions = gridmodel.predict(Xtest)
cm = confusion_matrix(Ytest, grid_predictions,labels=[0,1])
print("Confusion Matrix")
print(cm)
result=classification_report(Ytest, grid_predictions)
print(result)
rheat = classification_report(Ytest, grid_predictions,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Ytest,grid_predictions)
print("Accuracy:",result2)
#Applying Linear Kernel
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['linear']}
from sklearn.model_selection import GridSearchCV
gridmodel=GridSearchCV(SVC(), param_grid,refit=True,verbose=3)
gridmodel.fit(Xtrain,Ytrain)
gridmodel.best_params_
gridmodel.best_estimator_
grid_predictions = gridmodel.predict(Xtest)
cm = confusion_matrix(Ytest, grid_predictions,labels=[0,1])
print("Confusion Matrix")
print(cm)
result=classification_report(Ytest, grid_predictions)
print(result)
rheat = classification_report(Ytest, grid_predictions,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Ytest,grid_predictions)
print("Accuracy:",result2)

#Applying polynomial Kernel
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['poly']}
from sklearn.model_selection import GridSearchCV
gridmodel=GridSearchCV(SVC(), param_grid,refit=True,verbose=3)
gridmodel.fit(Xtrain,Ytrain)
gridmodel.best_params_
gridmodel.best_estimator_
grid_predictions = gridmodel.predict(Xtest)
cm = confusion_matrix(Ytest, grid_predictions,labels=[0,1])
print("Confusion Matrix")
print(cm)
result=classification_report(Ytest, grid_predictions)
print(result)
rheat = classification_report(Ytest, grid_predictions,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Ytest,grid_predictions)
print("Accuracy:",result2)
# =============================================================================
# Y_array=np.asarray(Ytrain.size_category)
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# Xtrain2 = pca.fit_transform(Xtrain)
# model.fit(Xtrain2, Ytrain)
# X_array=np.asarray(Xtrain,dtype=int)
# from mlxtend.plotting import plot_decision_regions
# # Plot Decision Region using mlxtend's awesome plotting function
# plot_decision_regions(X=X_array, 
#                       y=Y_array,
#                       clf=model, 
#                       legend=2)
# 
# # Update plot object with X/Y axis labels and Figure Title
# plt.title('SVM Decision Region Boundary', size=16)
# =============================================================================

