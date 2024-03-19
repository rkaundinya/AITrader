#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 08:23:15 2023

@author: andrewriznyk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv('../data/tradeAndNonTradePrompts.csv', delimiter = ',')

#nltk.download('stopwords')
corpus = []

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset["Trade"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

"""
corpus = []
# Keep Stop Words
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset["Trade"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = ' '.join(review)
    corpus.append(review)
"""
    
# Create Bag of Words

cv = CountVectorizer(max_features = 5500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#  Naive Bayes

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred_NB = classifier.predict(X_test)

cm_NB = confusion_matrix(y_test, y_pred_NB) 


TP_NB = cm_NB[1][1]
TN_NB = cm_NB[0][0]
FP_NB = cm_NB[1][0]
FN_NB = cm_NB[0][1]

'''
if TP_NB == 0:
    TP_NB = 1
if TN_NB == 0:
    TN_NB = 1
if FP_NB == 0:
    FP_NB = 1
if FN_NB == 0:
    FN_NB = 1   
'''
    
print("")
print("Naive Bayes to determine if text is a trade or not")
print("")

Accuracy_NB = (TP_NB + TN_NB) / (TP_NB + TN_NB + FP_NB + FN_NB) 
Precision_NB = TP_NB / (TP_NB + FP_NB)
Recall_NB = TP_NB / (TP_NB + FN_NB)
F1_Score_NB = 2 * Precision_NB * Recall_NB / (Precision_NB + Recall_NB)

print("Accuracy :" + str(Accuracy_NB))
print("Precision :" + str(Precision_NB))
print("Recall :" + str(Recall_NB))
print("F1 :" + str(F1_Score_NB))


# Decision Tree

classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred_DT = classifier.predict(X_test)

cm_DT = confusion_matrix(y_test, y_pred_DT) 

TP_DT = cm_DT[1][1]
TN_DT = cm_DT[0][0]
FP_DT = cm_DT[1][0]
FN_DT = cm_DT[0][1]

'''
if TP_DT == 0:
    TP_DT = 1
if TN_DT == 0:
    TN_DT = 1
if FP_DT == 0:
    FP_DT = 1
if FN_DT == 0:
    FN_DT = 1    
'''
   
print("")
print("Decision Tree to determine if text is a trade or not")
print("")
Accuracy_DT = (TP_DT + TN_DT) / (TP_DT + TN_DT + FP_DT + FN_DT) 
Precision_DT = TP_DT / (TP_DT + FP_DT)
Recall_DT = TP_DT / (TP_DT + FN_DT)
F1_Score_DT = 2 * Precision_DT * Recall_DT / (Precision_DT + Recall_DT)

print("Accuracy :" + str(Accuracy_DT))
print("Precision :" + str(Precision_DT))
print("Recall :" + str(Recall_DT))
print("F1 :" + str(F1_Score_DT))


# Random Forrest

classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred_RF = classifier.predict(X_test)

cm_RF = confusion_matrix(y_test, y_pred_RF)

TP_RF = cm_RF[1][1]
TN_RF = cm_RF[0][0]
FP_RF = cm_RF[1][0]
FN_RF = cm_RF[0][1]

'''
if TP_RF == 0:
    TP_RF = 1
if TN_RF == 0:
    TN_RF = 1
if FP_RF == 0:
    FP_RF = 1
if FN_RF == 0:
    FN_RF = 1    
'''

print("")
print("Random Forest to determine if text is a trade or not")
print("")
Accuracy_RF = (TP_RF + TN_RF) / (TP_RF + TN_RF + FP_RF + FN_RF)
Precision_RF = TP_RF / (TP_RF + FP_RF)
Recall_RF = TP_RF / (TP_RF + FN_RF)
F1_Score_RF = 2 * Precision_RF * Recall_RF / (Precision_RF + Recall_RF)

print("Accuracy :" + str(Accuracy_RF))
print("Precision :" + str(Precision_RF))
print("Recall :" + str(Recall_RF))
print("F1 :" + str(F1_Score_RF))