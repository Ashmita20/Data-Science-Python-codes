# -*- coding: utf-8 -*-
"""
Spyder Editor
# created by Ashmita

This is a temporary script file.
"""
# numpy library is used for mathematical functions
import numpy as np
# for plotting matplotlib is used.pyplot is primarily for working interactively. The exceptions are the pyplot commands figure(), subplot(), subplots(), and savefig(), which can greatly simplify scripting
import matplotlib.pyplot as mp
import pandas as pd


headernames = ("Undergrad","Marital.Status","Taxable.Income","City.Population","Work.Experience","Urban")
dataset = pd.read_csv("E://Datasets//Random Forest//Fraud_check.csv")
dataset.head(10)
result = dataset.dtypes
print(result)
Undergrad = {'YES':1,'NO':0}
dataset.Undergrad = [Undergrad[item] for item in dataset.Undergrad] 
dataset.head(10)
result = dataset.dtypes
print(result)
Urban = {'YES':1,'NO':0}
dataset.Urban = [Urban[item] for item in dataset.Urban] 
dataset.head(10)
result = dataset.dtypes
print(result)
Martial_Status = {'Single' :0,'Married':2,'Divorced':1}
dataset.MaritalStatus = [Martial_Status[item] for item in dataset.MaritalStatus]
# below code is not converting datatype so commented
# dataset.MaritalStatus[dataset.MaritalStatus == 'Single'] = 0
# dataset.MaritalStatus[dataset.MaritalStatus == 'Divorced'] = 1
# dataset.MaritalStatus[dataset.MaritalStatus == 'Married'] = 2
dataset.head(10)
result = dataset.dtypes
print(result)
dataset.TaxableIncome= np.where(dataset.TaxableIncome <= 30000,"Risky","Good")
dataset.head(10)
result = dataset.dtypes
print(result)
dataset['TaxableIncome'].value_counts()
print(dataset) 
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 4].values
X=dataset[['Undergrad','MaritalStatus','WorkExperience','Urban']]#Features
Y= dataset[['TaxableIncome']]
#X = dataset.iloc[:,0:6].values
#y = dataset.iloc[:,0:6].values
#y.tail()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
from sklearn.ensemble import RandomForestClassifier
# FOR 1000 TREES
classifier = RandomForestClassifier(n_estimators = 1000)
model = classifier.fit(X_train, np.ravel(Y_train))
y_pred = classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
import seaborn as sns
# Confusion matrix map
cm = pd.DataFrame(confusion_matrix(Y_test, y_pred))
print("Confusion Matrix:")
sns.heatmap(cm, annot=True)
result1 = classification_report(Y_test, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(result1).iloc[:-1, :].T, annot=True)
Y_array=np.asarray(Y_test.TaxableIncome)
pd.crosstab(Y_array, y_pred,rownames=["Actual Result"],colnames=["Predicted Result"])
# fOR 500 TREES
classifier = RandomForestClassifier(n_estimators = 500)
model = classifier.fit(X_train, np.ravel(Y_train))
y_pred = classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
import seaborn as sns
# Confusion matrix map
cm = pd.DataFrame(confusion_matrix(Y_test, y_pred))
print("Confusion Matrix:")
sns.heatmap(cm, annot=True)
result1 = classification_report(Y_test, y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(result1).iloc[:-1, :].T, annot=True)
Y_array=np.asarray(Y_test.TaxableIncome)
pd.crosstab(Y_array, y_pred,rownames=["Actual Result"],colnames=["Predicted Result"])
feature_cols = ['Undergrad','MaritalStatus','WorkExperience','Urban']#Features]
label=["TaxableIncome"]
X = dataset[feature_cols] # Features
Y = dataset[label] # Target variable
from sklearn.tree import DecisionTreeClassifier
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
Y_array=np.asarray(Y_test.TaxableIncome)
pd.crosstab(Y_array, Dpreds,rownames=["Actual Result"],colnames=["Predicted Result"])
from sklearn.tree import DecisionTreeClassifier
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
                special_characters=True, feature_names = feature_cols,class_names=['Good','Risky'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph[0].write_png('fraud.png')
Image(graph[0].create_png())

# # Import tools needed for visualization
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = classifier.estimators_[5]
# # Import tools needed for visualization
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = classifier.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')
# mp.plot(y_pred)

