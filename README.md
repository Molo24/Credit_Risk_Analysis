# Credit_Risk_Analysis

## Overview
The purpose of this project was to leverage machin learning (ML) tools and alorithms to study credit risk. The data used was from LendingClub, a peer-to-peer lending services company. This dataset was analyzed following Supervised Machine Learning techniques, including: regression, classifications, resampling and training among others.

The dataset was oversampled using the ```RandomOverSampler```, ```SMOTE``` algorithms, undersampled using the ```ClusterCentroids``` algorithm and over/undersampled using the ```SMOTEENN``` alorithm. Finally, ```BalancedRandomForestClassifier``` and ```EasyEnsembleClassifier``` machine learning models were used to reduce bias and to predict credit risk. 

Tools used: Python, Jupyter Notebook, ```imbalanced-learn``` Python libary and the ```scikit-learn``` Python library.

## Analysis
### Data Preparation
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

### Part 1: Resampling (Over and Under) models to Predict Credit Risk
#### 1-A: Oversample the data using the naive Random Oversampling algorithm.
![Part 1a](https://user-images.githubusercontent.com/89284280/147179346-8d2f12c1-f3ab-4fd7-93f6-31746801e233.jpg)

#### 1-B: Oversample the data using the SMOTE algorithm.
![Part 1b](https://user-images.githubusercontent.com/89284280/147179260-f227b12c-989e-4dc8-afb2-31583c71ddc2.jpg)

#### 1-C: Undersample the data using the Cluster Centroids algorithm.
![Part 1c](https://user-images.githubusercontent.com/89284280/147179566-3afc1109-d674-41e9-9404-db2fa70d2a99.jpg)

### Part 2: Use a combinatorial approach of over- and undersampling to Predict Credit Risk
#### Resample the data using the SMOTEENN algorithm
![Part 2](https://user-images.githubusercontent.com/89284280/147179900-effcfd60-681a-43cb-b35a-94dc9634afb3.jpg)

### Part 3: Use Ensemble Classifiers to Predict Credit Risk
#### 3-A: Using ```BalancedRandomForestClassifier``` Model
![Part 3a](https://user-images.githubusercontent.com/89284280/147180785-ab352c5e-f629-4ce1-9d33-9b8414776f85.jpg)

#### 3-B Using ```EasyEnsembleClassifier``` Model
![Part 3b](https://user-images.githubusercontent.com/89284280/147180796-56ea8ef3-84b6-40e2-96be-2c0b3c9f4be0.jpg)

## Results

|    | Random Oversampling | SMOTE | Cluster Centroids | SMOTEENN | BalancedRandomForestClassifier | EasyEnsembleClassifier |
|----|---------------------|-------|-------------------|----------|--------------------------------|------------------------|
|Balanced Accuracy Score|0.638|0.661|0.544|0.649|0.784|0.933|
|Precision (high_risk)|0.01|0.01|0.01|0.01|0.03|0.09|
|Precision (low_risk)|1.00|1.00|1.00|1.00|1.00|1.00|
|Recall (high_risk)|0.66|0.63|0.69|0.72|0.69|0.92|
|Recall (low_risk)|0.61|0.69|0.40|0.57|0.88|0.94|

<br>

## Summary
### Accuracy
Accuracy considers how often the classifier is correct in the model.

The EasyEnsembleClassifier had the highest accuracy at 93.3%

### Precision
Precision evaluates the reliability of a positive outcome. A high precision is indicative of a low amount of false positives.

The EasyEnsembleClassifier had the highest precision for the ```high_risk``` group, albeit incredibly low and with very little difference between the models. Further, the precision of the ```low_risk``` models were all the same. 

### Recall
Recall evaluates the classifier and whether or not it can find all the positive samples. In other words, a large number of false negatives is indicative of a low recall.

The EasyEnsembleClassifier had the highest recall for both ```high_risk``` and ```low_risk``` groups.

### Overall
The EasyEnsembleClassifier did the best job at evaluating credit risk of the applicants in the dataset.
