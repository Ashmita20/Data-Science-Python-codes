# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:43:51 2020

@author: CHARU NANDAN
"""


# import pandas library 
import pandas as pd 
file="E:\\Datasets\\Recommendation system\\book.csv" 
# Get the data 
#column_names = ['serial_id', 'user_id', 'book_title','rating'] 
import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
  
df = pd.read_csv(file,encoding='ISO-8859-1')
  
# Check the head of the data 
df.head()
df.drop(columns=['Serial_id'])
# Creating the groupby dictionary  
groupby_dict = {'User.ID':'Column 1', 
                'Book.Title':'Column 1'} 
df.groupby('Book.Title')['Book.Rating'].mean().sort_values(ascending=False).head()
#df.groupby('Book.Title')['Book.Rating'].mean().sort_values(ascending=False).head()  
ratings = pd.DataFrame(df.groupby('Book.Title')['Book.Rating'].mean())  
ratings.head()
ratings.sort_values(by='Book.Rating',ascending=False)  
ratings['num of ratings'] = pd.DataFrame(df.groupby('Book.Title')['Book.Rating'].count()) 
ratings=ratings.sort_values(by='num of ratings',ascending=False)   
ratings.head()

import matplotlib.pyplot as plt 
import seaborn as sns 
 # plot graph of 'num of ratings column' 
plt.figure(figsize =(10, 4)) 
  
ratings['num of ratings'].hist(bins = 70) 

split_value = int(len(df) * 0.80)
train_data = df[:split_value]
test_data = df[split_value:]
plt.figure(figsize = (12, 8))
ax = sns.countplot(x="Book.Rating", data=train_data)
ax.set_yticklabels(ratings['num of ratings'])
plt.tick_params(labelsize = 15)
plt.title("Count Ratings in train data", fontsize = 20)
plt.xlabel("Ratings", fontsize = 20)
plt.ylabel("Number of Ratings", fontsize = 20)
plt.show()


import numpy as np
total_users = len(np.unique(df["User.ID"]))
train_users = len(np.unique(train_data["User.ID"]))
uncommonUsers = total_users - train_users
                  
print("Total no. of Users = {}".format(total_users))
print("No. of Users in train data= {}".format(train_users))
print("No. of Users not present in train data = {}({}%)".format(uncommonUsers, np.round((uncommonUsers/total_users)*100), 2))

total_books = len(np.unique(df["Book.Title"]))
train_books = len(np.unique(train_data["Book.Title"]))
uncommonBook = total_books - train_books
print("Total no. of books = {}".format(total_books))
print("No. of books in train data= {}".format(train_books))
print("No. of books not present in train data = {}({}%)".format(uncommonBook, np.round((uncommonBook/total_books)*100), 2))

from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
len(train_data['Book.Title'].unique())
#train_data['Book.Title']=le.fit_transform(train_data['Book.Title'])
#test_data['Book.Title']=le.fit_transform(test_data['Book.Title'])
x_train=train_data[["User.ID","Book.Title"]]
x_test=test_data[["User.ID","Book.Title"]]
x_train['Book.Title']=le.fit_transform(x_train['Book.Title'])
x_test['Book.Title']=le.fit_transform(x_test['Book.Title'])
y_train=train_data["Book.Rating"]
y_test=test_data["Book.Rating"]

plt.figure(figsize = (12, 8))
ax = sns.countplot(x="Book.Title", data=x_train)
bx = sns.countplot(y="Book.Rating", data=train_data)
plt.tick_params(labelsize = 15)
plt.title("Count Ratings in train data", fontsize = 20)
plt.xlabel("Book Title", fontsize = 20)
plt.ylabel("Number of Ratings", fontsize = 20)
plt.show()
                  
from sklearn.metrics import mean_squared_error
import xgboost as xgb
clf = xgb.XGBRegressor(n_estimators = 100, silent = False, n_jobs  = 10)
clf.fit(x_train, y_train)
y_pred_test = clf.predict(x_test)
def error_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse
rmse_test = error_metrics(y_test, y_pred_test)
print("RMSE = {}".format(rmse_test))
def plot_importance(model, clf):
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_axes([0,0,1,1])
    model.plot_importance(clf, ax = ax, height = 0.3)
    plt.xlabel("F Score", fontsize = 20)
    plt.ylabel("Features", fontsize = 20)
    plt.title("Feature Importance", fontsize = 20)
    plt.tick_params(labelsize = 15)
    
    plt.show()
    plot_importance(xgb, clf)
ratings=ratings.sort_values(by='Book.Rating',ascending=False)   
ratings.head()
