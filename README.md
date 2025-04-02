# Classifying animals using unsupervised Clustering models
![image](https://github.com/user-attachments/assets/6257f8c7-063c-40c2-8035-baf0cd53fb2c)


## Introduction

If we are given a set of characteristics relative to an animal, are we able to exactly determine which class it corresponds? We could probably check the presence of unique class features, like the presence of milk-producing glands in the case of mammals or fins in fishes. Maybe we could see the overall most important features and then make a decision.

In this project, we use an unsupervised approach to classify each animal from a dataset based on the presence of different features. The results shown below compare between different clustering methods the level of accuracy given a selected animal class.

## Some notes

- This repository serves for the development of the project respective to the python module of the Coding for DS and DM course of the DSE Masters program at Unimi (First Trimester).
- Any information regarding the status of the project, changes, features, analysis and also polishing of the readme file will be documented here.
- **Student name**: Franco Faúndez Cabion


## ❔Confusion and more confusion

To get a better understanding of the results and also to standardize measured results, we applied the concept of confusion matrix for each class. This concept is often used in binary classification problems, but we can adapt our data in such a way that we can access this useful measuring tool. For each label (either predicted or true) of a particular class X we can create a new variable that has the value 1 when it corresponds to the class, or 0 if it doesn't. Then we compute the confusion matrix and this way, we can get our desired metrics

![image](https://github.com/user-attachments/assets/6a878690-97db-4d81-975e-e646a6f8f77b)

All of the metrics used are computed using the values of the confusion matrix. To have a better understanding of these metrics, here is a brief explanation:

Accuracy: of total predictions, what percentage is right?

Precision: of the positives predicted, what percentage is truly positive?

Specificity: how well is the model at predicting negative results?

Recall: how well is the model at predicting positive results?

F1: tells us how balanced accuracy and precision is. Do we have a lower or higher trade-off between these two metrics?

## Implementation using Streamlit

There's an implementation using Streamlit. Running the "dashboard_sl.py" using the files inside the data/description_clusters folder, will show the results on the classification and also performance of models.


