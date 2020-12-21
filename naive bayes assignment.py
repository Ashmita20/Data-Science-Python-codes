# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:44:23 2020

@author: Ashmita
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#from sklearn.model_selection import train_test_split

dataset_train = pd.read_csv("E://Datasets//Naive Bayes//SalaryData_Train.csv")
dataset_test = pd.read_csv("E://Datasets//Naive Bayes//SalaryData_Test.csv")

dataset_train.isnull().sum()
dataset_test.isnull().sum()

result = dataset_train.dtypes
print(result)
sns.countplot(x="Salary",data=dataset_train)

sns.countplot(x="Salary",data=dataset_test)
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
dataset_train['Salary'].unique()
# =============================================================================
# dataset_train.Salary= np.where(dataset_train.Salary == " <=50K","Normal","Good")
# dataset_train.head(10)
# dataset_test.Salary= np.where(dataset_test.Salary == " <=50K","Normal","Good")
# dataset_test.head(10)
# 
# =============================================================================

# Converting string labels into numbers.
dataset_train['workclass']=le.fit_transform(dataset_train['workclass'])
dataset_test['workclass']=le.fit_transform(dataset_test['workclass'])
dataset_train['Salary']=le.fit_transform(dataset_train['Salary'])
dataset_test['Salary']=le.fit_transform(dataset_test['Salary'])
# =============================================================================
# dataset_train['workclass'].unique()
# workclass ={' State-gov':0, ' Self-emp-not-inc':1, ' Private':2, ' Federal-gov':3, ' Local-gov':4, ' Self-emp-inc':5, ' Without-pay':6}
# dataset_train.workclass = [workclass[item] for item in dataset_train.workclass] 
# dataset_train.head(10)
# dataset_test.workclass = [workclass[item] for item in dataset_test.workclass]
# dataset_test.head()

# =============================================================================
dataset_train['education'].unique()
dataset_train['education']=le.fit_transform(dataset_train['education'])
dataset_test['education']=le.fit_transform(dataset_test['education'])
# =============================================================================
# education ={' Bachelors':0, ' HS-grad':1, ' 11th':2, ' Masters':3, ' 9th':4,
#        ' Some-college':5, ' Assoc-acdm':6, ' 7th-8th':7, ' Doctorate':8,
#        ' Assoc-voc':9, ' Prof-school':10, ' 5th-6th':11, ' 10th':12, ' Preschool':13,
#        ' 12th':14, ' 1st-4th':15}
# dataset_train.education = [education[item] for item in dataset_train.education] 
# dataset_train.head(10)
# dataset_test.education = [education[item] for item in dataset_test.education]
# dataset_test.head()

# =============================================================================
dataset_train['occupation'].unique()
dataset_train['occupation']=le.fit_transform(dataset_train['occupation'])
dataset_test['occupation']=le.fit_transform(dataset_test['occupation'])
# =============================================================================
# occupation ={' Adm-clerical':0, ' Exec-managerial':1, ' Handlers-cleaners':2,
#        ' Prof-specialty':3, ' Other-service':4, ' Sales':5, ' Transport-moving':6,
#        ' Farming-fishing':7, ' Machine-op-inspct':8, ' Tech-support':9,
#        ' Craft-repair':10, ' Protective-serv':11, ' Armed-Forces':12,
#        ' Priv-house-serv':13}
# dataset_train.occupation = [occupation[item] for item in dataset_train.occupation] 
# dataset_train.head(10)
# dataset_test.occupation = [occupation[item] for item in dataset_test.occupation]
# dataset_test.head()
# =============================================================================
sns.barplot(x="Salary",y="education",data=dataset_train)
sns.barplot(x="Salary",y="workclass",data=dataset_train)
sns.barplot(x="Salary",y="occupation",data=dataset_train)
# dropping data which is not required
train =dataset_train[["workclass","education","occupation","capitalgain","capitalloss","hoursperweek","Salary"]]
X_train = train[["workclass","education","occupation","capitalgain","capitalloss","hoursperweek"]]
Y_train = train[["Salary"]]
test= dataset_test[["workclass","education","occupation","capitalgain","capitalloss","hoursperweek","Salary"]]
X_test = test[["workclass","education","occupation","capitalgain","capitalloss","hoursperweek"]]
Y_test = test[["Salary"]]

from sklearn.naive_bayes import CategoricalNB
model = CategoricalNB()
nb= model.fit(X_train,np.ravel(Y_train))
y_pred = model.predict(X_test)
print(y_pred)
print(nb)
Y_array=np.asarray(Y_test.Salary)
result = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
rheat = classification_report(Y_test, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
pd.crosstab(Y_array, y_pred,rownames=["Actual Result"],colnames=["Predicted Result"])


from mixed_naive_bayes import MixedNB
model = MixedNB(categorical_features=[0,1])
nb= model.fit(X_train,np.ravel(Y_train))
y_pred = model.predict(X_test)
print(y_pred)
print(nb)
Y_array=np.asarray(Y_test.Salary)
result = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
rheat = classification_report(Y_test, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
pd.crosstab(Y_array, y_pred,rownames=["Actual Result"],colnames=["Predicted Result"])


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
nb= model.fit(X_train,np.ravel(Y_train))
y_pred = model.predict(X_test)
print(y_pred)
print(nb)
Y_array=np.asarray(Y_test.Salary)
result = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
rheat = classification_report(Y_test, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
pd.crosstab(Y_array, y_pred,rownames=["Actual Result"],colnames=["Predicted Result"])

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
nb= model.fit(X_train,np.ravel(Y_train))
y_pred = model.predict(X_test)
print(y_pred)
print(nb)
Y_array=np.asarray(Y_test.Salary)
result = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
rheat = classification_report(Y_test, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
pd.crosstab(Y_array, y_pred,rownames=["Actual Result"],colnames=["Predicted Result"])

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
nb= model.fit(X_train,np.ravel(Y_train))
y_pred = model.predict(X_test)
print(y_pred)
print(nb)
Y_array=np.asarray(Y_test.Salary)
result = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
rheat = classification_report(Y_test, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(rheat).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
pd.crosstab(Y_array, y_pred,rownames=["Actual Result"],colnames=["Predicted Result"])
