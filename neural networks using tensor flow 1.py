# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:37:26 2020

@author: CHARU NANDAN
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
df =  pd.read_csv("E:\\Datasets\\Neural Networks\\forestfires.csv")
#creating labelEncoder
le = LabelEncoder()
df['month']=le.fit_transform(df['month'])
df['day']=le.fit_transform(df['day'])
df['size_category']=le.fit_transform(df['size_category'])
df.head(10)
df.columns
# print(len(df.columns))
# Taking neccessary columns only
X = df[df.columns[2:11]].values
y = df[df.columns[30]]
print(y)
def normalize_features(features):
    mu = np.mean(features,axis=0)
    std = np.std(features,axis=0)
    normalize_features = (features-mu)/std
    return(normalize_features)

# Define the encoder function.
def one_hot_encode(labels):
    n_labels = len(labels)
    #print(n_labels)
    n_unique_labels = len(np.unique(labels))
    #print("test1 " ,n_unique_labels)
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    #print("test2 " ,one_hot_encode)
    one_hot_encode[np.arange(n_labels), labels] = 1
    #print("test3 " ,np.arange(n_labels))
    #print("test 4",labels)
    #print("test 5",one_hot_encode)
    return one_hot_encode

def plot_points(features,labels):
    normal=np.where(labels==0)
    outliers=np.where(labels==1)
    fig= plt.figure(figsize=(10,8))
    plt.plot(features[normal ,0],features[normal ,1],'bx')
    plt.plot(features[outliers ,0],features[outliers ,1],'ro')
    plt.show()

Y=one_hot_encode(y)
print(X.shape)
print(Y) 
X = normalize_features(X)
plot_points(X,y)
# =============================================================================
# # Define the encoder function.
# def one_hot_encode(labels):
#     n_labels = len(labels)
#     n_unique_labels = len(np.unique(labels))
#     one_hot_encode = np.zeros((n_labels, n_unique_labels))
#     one_hot_encode[np.arange(n_labels), labels] = 1
#     return one_hot_encode
# =============================================================================
 
X[:,:]
# Shuffle the dataset to mix up the rows.
X, Y = shuffle(X, Y, random_state=1)
 
# Convert the dataset into train and test part
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)
 
# Inpect the shape of the training and testing.
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
 
# Define the important parameters and variable to work with the tensors
learning_rate = 0.7
training_epochs = 120
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
# As there are two values of size category so number of class =2 ie 0 & 1
n_class = 2

model_path = "E:\\Datasets\\Neural Networks\\forestfire"
 
# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 6
n_hidden_2 = 6
n_hidden_3 = 6
n_hidden_4 = 6

# =============================================================================
# n_hidden_5 = 7
# n_hidden_6 = 7
# =============================================================================

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 
x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None,n_class])
print(y_.shape)
 
 
# Define the model
def multilayer_perceptron(x, weights, biases):
 
    # Hidden layer with RELU activationsd
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
 
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
 
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)
 
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.tanh(layer_4)
    
   
    
# =============================================================================
#     # Hidden layer with RELU activation
#     
#      layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
#      layer_5 = tf.nn.tanh(layer_5)
#     # Hidden layer with RELU activation
#     layer_6 = tf.add(tf.matmul(layer_5, weights['h4']), biases['b4'])
#     layer_6 = tf.nn.relu(layer_6)
# =============================================================================
 
    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer
 
 
# Define the weights and the biases for each layer
 
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    
# =============================================================================
#     'h5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5])),
#     'h6': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_6])),
# =============================================================================
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    
# =============================================================================
#     'b5': tf.Variable(tf.truncated_normal([n_hidden_5])),
#     'b6': tf.Variable(tf.truncated_normal([n_hidden_6])),
# =============================================================================
    'out': tf.Variable(tf.truncated_normal([n_class]))
}
 
# Initialize all the variables
 
init = tf.global_variables_initializer()
 
saver = tf.train.Saver()
 
# Call your model defined
y = multilayer_perceptron(x, weights, biases)
print(y.shape)

# =============================================================================
# from ann_visualizer.visualize import ann_viz
# from keras.models import Sequential
# from keras.layers import Dense
# ann_viz(y)
# =============================================================================
 
 
# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
 
sess = tf.Session()
sess.run(init)
 
# Calculate the cost and the accuracy for each epoch
 
mse_history = []
accuracy_history = []
 
for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)
 
    print('epoch : ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)
 
save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)
 
#Plot Accuracy Graph
plt.plot(accuracy_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(mse_history)
plt.xlabel('Epoch')
plt.ylabel('cost')
plt.show()
 
# Print the final accuracy
 
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))
 
# Print the final mean square error
 
pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))
