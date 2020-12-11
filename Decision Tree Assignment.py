# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:27:16 2020

@author: user
"""


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataset = pd.read_csv("E:\\Datasets\\Decision Tree\\Company_Data.csv")
dataset.head()
# =============================================================================
# data['Sales'].unique()
# data.Sales.value_counts()
# colnames = list(data.columns)
# predictors = colnames[0:11]
# target = colnames[10]
# result = data.dtypes
# print(result)
# convert_dict = {'Sales': int(64),'ShelveLoc': int(64)}
# df= pd.DataFrame(data)
# df = df.astype(convert_dict)
# data.Urban[data.Urban == 'Yes'] = 1
# data.Urban[data.Urban == 'No'] = 0
# print(data)
# data.US[data.US == 'Yes'] = 1
# data.US[data.US == 'No'] = 0
# print(data)
# convert_dict = {'Urban': int(64),'US': int(64)}
# df= pd.DataFrame(data)
# df = df.astype(convert_dict) 
# result = df.dtype(convert_dict) 
# print(result)
# =============================================================================

#Converting string data in factor 
ShelveLoc = {'Good':2,'Medium':1,"Bad":0}
dataset.ShelveLoc = [ShelveLoc[item] for item in dataset.ShelveLoc] 
dataset.head(10)
result = dataset.dtypes

Urban = {'Yes':1,'No':0}
dataset.Urban = [Urban[item] for item in dataset.Urban] 
dataset.head(10)
result = dataset.dtypes
print(result)

US = {'Yes':1,'No':0}
dataset.US = [US[item] for item in dataset.US] 
dataset.head(10)
result = dataset.dtypes
print(result)

dataset.Sales= np.where(dataset.Sales <= 8,"Bad","Good")
dataset.head(10)
result = dataset.dtypes
print(result)
print(dataset)
# Splitting data into training and testing data set
X=dataset[["CompPrice","Price","Income","Advertising","Population","Age","Education","Urban","US","ShelveLoc"]]
Y=dataset[["Sales"]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

help(DecisionTreeClassifier)
#Random Forest Classification
classifier = RandomForestClassifier(n_estimators = 500)
rmodel = classifier.fit(X_train, np.ravel(Y_train))
y_pred = classifier.predict(X_test)

print(y_pred)
print(Y_test)
Y_array=np.asarray(Y_test.Sales)

result = confusion_matrix(Y_test, y_pred,labels=["Good","Bad"])
print("Confusion Matrix:")
print(result)

result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
# heat map for classification 
import seaborn as sns
# Confusion matrix map
cm = pd.DataFrame(confusion_matrix(Y_test, y_pred,labels=["Good","Bad"]))
print("Confusion Matrix:")
sns.heatmap(cm, annot=True)
result1 = classification_report(Y_test, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(result1).iloc[:-1, :].T, annot=True)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
pd.crosstab(Y_array, y_pred,rownames=["Actual Result"],colnames=["Predicted Result"])
# Splitting data into training and testing data set
feature_cols = ["CompPrice","Price","Income","Advertising","Population","Age","Education","Urban","US","ShelveLoc"]
label=["Sales"]
X = dataset[feature_cols] # Features
Y = dataset[label] # Target variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
# Decission Tree Classification using gini for gini impurity
Dclassifier = DecisionTreeClassifier(criterion="gini")
Dmodel= Dclassifier.fit(X_train, np.ravel(Y_train))
Dpreds = Dmodel.predict(X_test)
Dresult = confusion_matrix(Y_test, Dpreds)
print("Confusion Matrix:")
print(Dresult)
Dresult1 = classification_report(Y_test, Dpreds)
print("Classification Report:",)
print (Dresult1)
Dresult2 = accuracy_score(Y_test,Dpreds)
print("Accuracy:",Dresult2)
Y_array=np.asarray(Y_test.Sales)
pd.crosstab(Y_array, Dpreds,rownames=["Actual Result"],colnames=["Predicted Result"])
from six import StringIO
from IPython.display import Image  
import graphviz
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(Dmodel, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['Good','Bad'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('company_gini.png')
Image(graph.create_png())



Dclassifier = DecisionTreeClassifier(criterion="entropy")
Dmodel= Dclassifier.fit(X_train, np.ravel(Y_train))
Dpreds = Dmodel.predict(X_test)
Dresult = confusion_matrix(Y_test, Dpreds)
print("Confusion Matrix:")
print(Dresult)
Dresult1 = classification_report(Y_test, Dpreds)
print("Classification Report:",)
print (Dresult1)
Dresult2 = accuracy_score(Y_test,Dpreds)
print("Accuracy:",Dresult2)

from six import StringIO
from IPython.display import Image  
import graphviz
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(Dmodel, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['Good','Bad'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph[0].write_png('company_entropy.png')
Image(graph[0].create_png())


# =============================================================================
# import graphviz
# 
# dot = graphviz.Digraph(comment='The Round Table')
# 
# dot.node('A', 'King Arthur')
# dot.node('B', 'Sir Bedevere the Wise')
# dot.node('L', 'Sir Lancelot the Brave')
# dot.edges(['AB', 'AL'])
# dot.edge('B', 'L', constraint='false')
# 
# dot.render('FileName', view=True)
# =============================================================================

