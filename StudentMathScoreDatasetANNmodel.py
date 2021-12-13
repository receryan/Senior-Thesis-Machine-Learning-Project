# -*- coding: utf-8 -*-
"""
Rece Ryan
The database used can be found on the UCI Database
Titled: Student Performance

--------------------------------------------------------------------------------------------------------
Creating the Regression Model
Regression Model Chosen: Artificial Neural Network (ANN)
Reason: 
"""

#Importing the basic libraries necessary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Importing the dataset
dataset = pd.read_csv('student-mat-edited.csv')


#Splitting dataset into nonpredictive and predictive attributes
x = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , -1].values


#Importing and using ColumnTransformer and OneHotEncoder to encode the categorical data (17 columns encoded)
#This will expand the dataset based on how many attributes contain categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#use ranges here
columnT = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22])], remainder = 'passthrough')
x = np.array(columnT.fit_transform(x), dtype = np.float)


#Importing and using train_test_split to split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


#Importing and using StandardScaler to give all features equal weight
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Importing and using the ANN model
import keras
from keras.models import Sequential
from keras.layers import Dense

#Created regressor
regressor = Sequential()

#Added Input layer/first hidden layer, second hidden layer, and output layer
regressor.add(Dense(activation = "relu", input_dim = 58, units = 12, kernel_initializer = "uniform"))
regressor.add(Dense(activation = "relu", units = 8, kernel_initializer = "uniform"))
regressor.add(Dense(1))

#Fixed MAPE calculation error that was producing unreasonably large numbers
keras.backend.set_epsilon(1)

#Compiled the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitted to training set
regressor.fit(x_train, y_train, batch_size = 10, epochs = 100)

#Predicting results
y_pred = regressor.predict(x_test)


#Calculating MSE and RMSE
from sklearn.metrics import mean_squared_error
error3 = mean_squared_error(y_test, y_pred)
rmse3 = np.sqrt(error3)


#Visualizing Results
import seaborn as sb

sb.set_style("darkgrid")
sb.scatterplot(data = y_test, color = "black", label = "Final Grades Given")
sb.lineplot(data = y_pred, palette = "flare", label = "Predicted Grades")
plt.xlabel("Instance of Student in Subset", fontsize = 15)
plt.ylabel("Final Grade", fontsize = 15)


