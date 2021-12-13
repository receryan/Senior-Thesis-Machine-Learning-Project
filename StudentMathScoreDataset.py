# -*- coding: utf-8 -*-
"""
Rece Ryan
The database used can be found on the UCI Database
Titled: Student Performance

--------------------------------------------------------------------------------------------------------
Creating the Regression Model
Regression Model Chosen: Multiple Linear Regression
Reason: The dataset contains many attributes; we must flush out any that don't help us predict something
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


#Importing and using LinearRegression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predictor Variable
y_pred = regressor.predict(x_test)


#Calculate MSE and RMSE
from sklearn.metrics import mean_squared_error
error1 = mean_squared_error(y_test, y_pred)
rmse1 = np.sqrt(error1)

#Use p values to eliminate features
import statsmodels.api as sm

x = np.append(arr = np.ones((395,1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 44, 50, 56, 57, 58]]

regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())


#Visualizing the results using Seaborn

import seaborn as sb
    
sb.relplot(data = dataset, x = "G3", y = "age", hue = "G3")
plt.title('Final Grade vs. Student Ages')

sb.relplot(data = dataset, x = "G3", y = "famrel", hue = "G3")
plt.title('Final Grade vs. Family Relationships')

sb.relplot(data = dataset, x = "G3", y = "absences", hue = "G3")
plt.title('Final Grade vs. Number of Absences')

sb.relplot(data = dataset, x = "G3", y = "G1", hue = "G3")
plt.title('Final Grade vs. First Period Grade')

sb.relplot(data = dataset, x = "G3", y = "G2", hue = "G3")
plt.title('Final Grade vs Second Period Grade')
sb.set_style("dark")


plt.show()





