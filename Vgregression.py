import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#importing the dataset
Dataset = pd.read_csv('vgsales.csv')
#Label Encoding
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()

Dataset['Platform'] = number.fit_transform(Dataset['Platform'].astype('str'))
Dataset['Genre'] = number.fit_transform(Dataset['Genre'].astype('str'))
Dataset['Publisher'] = number.fit_transform(Dataset['Publisher'].astype('str'))

#extracting the feature vector and the dependant variable vector

columns = ["Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales"]


y = Dataset["Global_Sales"].values
X = Dataset[list(columns)].values

#importing the linear model library
from sklearn import linear_model

regr = linear_model.LinearRegression()
#importing the train test split library and splitting data into 80% for training 20% for testing
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#scaling the data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)
#fit the train data to the linear model
regr.fit(X_train, y_train)

#Printing Accuracy in our Linear model
Accuracy = regr.score(X_train, y_train)
print ("Accuracy in the training data with Linear Regression Model: ", Accuracy*100, "%")

accuracy = regr.score(X_test, y_test)
print ("Accuracy in the test data with Linear Regression model", accuracy*100, "%")
#Comparing the model predicted results vs the Test set
y_pred_Model1 = regr.predict(X_test)
y_pred_Model1


compare_Model1 = np.concatenate((y_pred_Model1.reshape(len(y_pred_Model1),1), y_test.reshape(len(y_test),1)),1)
compare_Model1
####*************************
#Using DecisionTreeRegressor : 
from sklearn.tree import DecisionTreeRegressor
DTR  = DecisionTreeRegressor()
DTR2 =DecisionTreeRegressor(min_samples_leaf=0.2)
DTR3 =DecisionTreeRegressor(min_samples_leaf=15)
DTR4 =DecisionTreeRegressor(min_samples_leaf=30)
DTR5 =DecisionTreeRegressor(min_samples_leaf=35)


DTR.fit(X_train, y_train)
DTR2.fit(X_train,y_train)
DTR3.fit(X_train,y_train)
DTR4.fit(X_train,y_train)
DTR5.fit(X_train,y_train)

#printing Accuracy in our DTR Model

Accuracy = DTR.score(X_train, y_train)
print ("Accuracy in the training data with Decision Tree Regression model before tuning  : ", Accuracy*100, "%")

accuracy = DTR.score(X_test, y_test)
print ("Accuracy in the test data with Decision Tree Regression model before tuining : ", accuracy*100, "%")
##########################
Accuracy2 = DTR2.score(X_train, y_train)
print ("\nAccuracy in the training data with Decision Tree Regression model After tuning with min_samples_leaf=3 : ", Accuracy2*100, "%")

accuracy2 = DTR2.score(X_test, y_test)
print ("Accuracy in the test data with Decision Tree Regression model After tuning with min_samples_leaf=3 : ", accuracy2*100, "%")
##########################
Accuracy3 = DTR3.score(X_train, y_train)
print ("\nAccuracy in the training data with Decision Tree Regression model After tuning with min_samples_leaf=15 : ", Accuracy3*100, "%")

accuracy3 = DTR3.score(X_test, y_test)
print ("Accuracy in the test data with Decision Tree Regression model After tuning with min_samples_leaf=15 : ", accuracy3*100, "%")
##########################
Accuracy4 = DTR4.score(X_train, y_train)
print ("\nAccuracy in the training data with Decision Tree Regression model After tuning with min_samples_leaf=30 : ", Accuracy4*100, "%")

accuracy4 = DTR4.score(X_test, y_test)
print ("Accuracy in the test data with Decision Tree Regression model After tuning with min_samples_leaf=30 : ", accuracy4*100, "%")
##########################
Accuracy5 = DTR5.score(X_train, y_train)
print ("\nAccuracy in the training data with Decision Tree Regression model After tuning with min_samples_leaf=35 : ", Accuracy5*100, "%")

accuracy5 = DTR5.score(X_test, y_test)
print ("Accuracy in the test data with Decision Tree Regression model After tuning with min_samples_leaf=35 : ", accuracy5*100, "%")
##########################

#Comparing the model predicted results vs the Test set
y_pred_Model2 = DTR.predict(X_test)
y_pred_Model2

y_pred_Model2_Tuned = DTR5.predict(X_test)
y_pred_Model2_Tuned

compare_Model2 = np.concatenate((y_pred_Model2.reshape(len(y_pred_Model2),1), y_test.reshape(len(y_test),1)),1)
compare_Model2

compare_Model2_Tuned = np.concatenate((y_pred_Model2_Tuned.reshape(len(y_pred_Model2_Tuned),1), y_test.reshape(len(y_test),1)),1)
compare_Model2_Tuned