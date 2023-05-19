#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:05:16 2023

@author: shreiyavenkatesan
"""
#%%
#importing required libraries for data analysis
import pandas as pd  
import numpy as np 
#importing required libraries for data visualization 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split 
#%%
#LOADING THE DATA
data = pd.read_csv('/Users/shreiyavenkatesan/Desktop/assignment1_winequality.csv')
data.head() 
data.info()
#%%
#SUMMARY STATISTICS AND DATA ANALYSIS
summary = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].describe()
#%%
# VISUALISING DATA TO FIND OUTLIERS
#A) SCATTER PLOT

data.plot(x='fixed acidity', y='quality', style='o')
data.plot(x='volatile acidity', y='quality', style='o')
data.plot(x='citric acid', y='quality', style='o')
data.plot(x='residual sugar', y='quality', style='o')
data.plot(x='chlorides', y='quality', style='o')
data.plot(x='free sulfur dioxide', y='quality', style='o')
data.plot(x='total sulfur dioxide', y='quality', style='o')
data.plot(x='density', y='quality', style='o')
data.plot(x='pH', y='quality', style='o')
data.plot(x='sulphates', y='quality', style='o')
data.plot(x='alcohol', y='quality', style='o')

#%%
# B) DENSITY PLOT
plt.figure(figsize = [20,10])
cols = data.columns
cnt = 1
for col in cols:
  plt.subplot(4,3,cnt)
  sns.distplot(data[col],hist_kws=dict(edgecolor="k", linewidth=1,color='lightblue'),color='red')
  cnt+=1
plt.tight_layout()
plt.show()
#%%
# C) BOX PLOT
for col in data.columns[0:12]:
    sns.boxplot(x='quality', y=col, data=data,palette='GnBu_d')
    plt.show()
#%%
#DROPPING THE EXTREME OUTLIERS AND NA VALUES 
data.drop(data[data['fixed acidity'] == -999].index, inplace = True)
data.drop(data[data['density'] == -9].index, inplace = True)
data.drop(data[data['pH'] == 31.3].index, inplace = True)
data.drop(data[data['chlorides'] == 11.2].index, inplace = True)
data.drop(data[data['total sulfur dioxide'] > 150].index, inplace = True)
data.drop(data[data['volatile acidity'] > 1].index, inplace = True)
data.drop(data[data['alcohol'] > 13].index, inplace = True)
data.drop(data[data['sulphates'] > 1].index, inplace = True)
data.dropna(inplace=True)
#%%
# CHECKING THE DATA AFTER CLEANING
data.info()
summary1 = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].describe()
data.isnull().sum()

#%%
#SCATTER PLOT OF CLEAN DATA
data.plot(x='fixed acidity', y='quality', style='o')
data.plot(x='volatile acidity', y='quality', style='o')
data.plot(x='citric acid', y='quality', style='o')
data.plot(x='residual sugar', y='quality', style='o')
data.plot(x='chlorides', y='quality', style='o')
data.plot(x='free sulfur dioxide', y='quality', style='o')
data.plot(x='total sulfur dioxide', y='quality', style='o')
data.plot(x='density', y='quality', style='o')
data.plot(x='pH', y='quality', style='o')
data.plot(x='sulphates', y='quality', style='o')
data.plot(x='alcohol', y='quality', style='o')
#%%
# DENSITY PLOT
plt.figure(figsize = [20,10])
cols = data.columns
cnt = 1
for col in cols:
  plt.subplot(4,3,cnt)
  sns.distplot(data[col],hist_kws=dict(edgecolor="k", linewidth=1,color='lightblue'),color='red')
  cnt+=1
plt.tight_layout()
plt.show()

#%%
# BOX PLOT 
for col in data.columns[0:12]:
    sns.boxplot(x='quality', y=col, data=data,palette='GnBu_d')
    plt.show()

#%%
#CORRELATION MATRIX FOR FEATURE SELECTION
plt.figure(figsize = (30,30))
sns.heatmap(data.corr(),annot=True, cmap= 'rocket')

#%%
#A BAR GRAPH TO SHOW THE FEATURES THAT ARE HIGHLY CORRELATED WITH QUALITY
plt.figure(figsize=(15,15))
data.corr()['quality'].apply(lambda x: abs(x)).sort_values(ascending=False).iloc[1:11][::-1].plot(kind='barh',color=('darkgreen','maroon')) 
plt.title("CORRELATED FEATURES", size=20, pad=26)
plt.xlabel("Correlation Coefficients")
plt.ylabel("Features")
#%%

#SORTING THE "QUALITY" INTO TWO GROUPS:LESS THAN 7 AND GREATER THAN 7
data['quality'] = [1 if x >=7 else 0 for x in data.quality]
#%%
# REMOVING THE FEAUTURE "RESIDUAL SUGAR" BECAUSE IT IS THE LEAST CORRELATED WITH QUALITY

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides',
'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
'sulphates', 'alcohol']
label = ['quality']
x = data.loc[:,features]
y = data.loc[:,['quality']]


#%%
##SPLITTING THE DATA INTO TEST AND TRAIN : 80% AND 20% RESPECTIVELY
xtrain, xtest, ytrain, ytest = train_test_split(x, y , 
                          random_state=42,
                          train_size=0.8,shuffle=(True))
#%%
##STANDARDISING THE FEATURES
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
xtrain_scaled=scaler.fit_transform(xtrain)
xtest_scaled=scaler.transform(xtest)
#%%

## K NEAREST NEIGHBOR CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

model2 = KNeighborsClassifier(n_neighbors=4)
model2.fit(xtrain,ytrain)
pred2 = model2.predict(xtest)
y_prob=model2.predict_proba(xtest)
mat2 = confusion_matrix(pred2, ytest)
names = np.unique(pred2)
mat2 = pd.DataFrame(mat2, index=np.unique(ytest), columns=np.unique(pred2))
mat2.index.name = 'Actual'
mat2.columns.name = 'Predicted'
sns.heatmap(mat2, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names,cmap="YlGnBu")
accuracy=np.round(round(accuracy_score(ytest,pred2),3)*100,2)
precision=np.round(round(precision_score(ytest,pred2,average='weighted'),3)*100,2)
sensitivity =np.round(round(recall_score(ytest,pred2,average='weighted'),3)*100,2)
print(f'Accuracy of the model: {accuracy}%')
print(f'Precision Score of the model: {precision}%')
print(f'Sensitivity of the model: {sensitivity}%')
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest,pred2)
print(mse)
#%%
##GAUSSIAN NAIVE BAYES CLASSIFIER
from sklearn.naive_bayes import GaussianNB
model1=GaussianNB()
model1.fit(xtrain,ytrain)
pred1 = model1.predict(xtest)
mat1 = confusion_matrix(pred1, ytest)
names = np.unique(pred1)
mat1 = pd.DataFrame(mat1, index=np.unique(ytest), columns=np.unique(pred2))
mat1.index.name = 'Actual'
mat1.columns.name = 'Predicted'
sns.heatmap(mat1, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names,cmap="twilight")
accuracy1=np.round(round(accuracy_score(ytest,pred1),3)*100,2)
precision1=np.round(round(precision_score(ytest,pred1,average='weighted'),3)*100,2)
sensitivity1 =np.round(round(recall_score(ytest,pred1,average='weighted'),3)*100,2)
print(f'Accuracy of the model: {accuracy1}%')
print(f'Precision Score of the model: {precision1}%')
print(f'Sensitivity of the model: {sensitivity1}%')
from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(ytest,pred1)
print(mse)

#%%
## BERNOULLI NAIVE BAYES CLASSIFIER
model = BernoulliNB()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
accuracyBN=np.round(round(accuracy_score(ytest,pred),3)*100,2)
precisionBN=np.round(round(precision_score(ytest,pred,average='weighted'),3)*100,2)
sensitivityBN =np.round(round(recall_score(ytest,pred,average='weighted'),3)*100,2)
print(f'Accuracy of the model: {accuracyBN}%')
print(f'Precision Score of the model: {precisionBN}%')
print(f'Sensitivity of the model: {sensitivityBN}%')
from sklearn.metrics import mean_squared_error
mseBN= mean_squared_error(ytest,pred)
print(mse)

