# Credit_Risk_Analysis

## Overview
The purpose of this project was to leverage machin learning (ML) tools and alorithms to study credit risk. The data used was from LendingClub, a peer-to-peer lending services company. This dataset was analyzed following Supervised Machine Learning techniques, including: regression, classifications, resampling and training among others.

The dataset was oversampled using the ```RandomOverSampler```, ```SMOTE``` algorithms, undersampled using the ```ClusterCentroids``` algorithm and over/undersampled using the ```SMOTEENN``` alorithm. Finally, ```BalancedRandomForestClassifier``` and ```EasyEnsembleClassifier``` machine learning models were used to reduce bias and to predict credit risk. 

Tools used: Python, Jupyter Notebook, ```imbalanced-learn``` Python libary and the ```scikit-learn``` Python library.

## Analysis
### Data Preparation
Using ``imbalanced-learn``` and ```scikit-learn``` libraies to evaluate three machine learning libraries by resampling to determine best at predicting credit risk.

Prior to any analysis beginning, the data set must be prepared. This was done using Panda Dataframes by splitting the data into separate Training and Testing dataframes.
```
# Create our features
X = df_encoded.drop(columns="loan_status")

# Create our target
y = df_encoded["loan_status"]
```
![X df](https://user-images.githubusercontent.com/89284280/147178331-2b55ada2-b6d3-457f-95a0-b778a3ee593a.JPG)

![y df](https://user-images.githubusercontent.com/89284280/147178337-8f6bd16a-ab39-4993-928d-2f20d4cdbdef.JPG)

Then, using ```scikit-learn``` the data were split into the X train/test and y train/test groups:
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

### Part 1: Resampling (Over and Under) Models to Predict Credit Risk
#### 1-A: Oversample the data using the naive Random Oversampling algorithm.
![Part 1a](https://user-images.githubusercontent.com/89284280/147179346-8d2f12c1-f3ab-4fd7-93f6-31746801e233.jpg)

#### 1-B: Oversample the data using the SMOTE algorithm.
![Part 1b](https://user-images.githubusercontent.com/89284280/147179260-f227b12c-989e-4dc8-afb2-31583c71ddc2.jpg)

#### 1-C: Undersample the data using the Cluster Centroids algorithm.
![Part 1c](https://user-images.githubusercontent.com/89284280/147179566-3afc1109-d674-41e9-9404-db2fa70d2a99.jpg)

### Part 2: Use a combinatorial approach of over- and undersampling
#### Resample the data using the SMOTEENN algorithm
![Part 2](https://user-images.githubusercontent.com/89284280/147179900-effcfd60-681a-43cb-b35a-94dc9634afb3.jpg)

### Part 2: Use a combinatorial approach of over- and undersampling
