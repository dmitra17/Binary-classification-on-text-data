#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:19:22 2017

@authors: Group-8

Script Name: SCP_05_KNN_Model.py

Script Description : This script loads the pre-processed files, split the data into 80-20% training-test set
                     and applies K-Nearest Neighbour model. It also displays the Accuracy, Precision, Recall and F1 Score.
                     
"""

#Import Libraries
from sklearn.cross_validation import train_test_split
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import operator
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import time


#FilePath
FILEPATH = "/Users/Avirup/Desktop/Fall, 2017/Introduction to Data Mining/Projects/Final Project/"

#Features file name
FEATURESET_FILENAME = "features_reviews_Clothing_Shoes_and_Jewelry.pickle"

#File containing the converted DataFrame
DATAFRAME_FILENAME = "reviews_Clothing_Shoes_and_Jewelry.pickle"

#Value of K
K = 3

#Number of important features to be retained
NUM_FEATURES = 10


#Function to compute euclidean distance
def fun_calc_euclideanDist(instance1, instance2):
    distance = np.array(np.sqrt(np.sum(np.power((instance1[:,:instance1.shape[1]-1] 
                                                 - instance2[:,:instance1.shape[1]-1]), 2), axis=1)))
    return distance
 

#Function to fetch k nearest neighbours based on distance metric
def fun_get_kNeighbors(trainingSet, test_vector, k):
    distances = []
    dist = fun_calc_euclideanDist(test_vector, trainingSet)
    distances = zip(np.array(trainingSet), dist)
    distances.sort(key=operator.itemgetter(1))
    neighbors = [x for x,_ in distances[:k]]
    return neighbors


#Function to predict class label based on k-nearest neigbours
def fun_getPrediction(neighbors):
    Votes = {}
    Votes = Counter([x[-1] for x in neighbors])
    return max(Votes, key=Votes.get)

#Check the start time
start_time = time.time()

#Load the pre-processed pickle files 
if os.path.exists(FILEPATH + FEATURESET_FILENAME) and os.path.exists(FILEPATH + FEATURESET_FILENAME):
    print ("Loading the file containing features...\n")
    features = pd.read_pickle(FILEPATH + FEATURESET_FILENAME)
    print ("Feature Set Pickle file loaded.\n")
    
    print ("Loading file containing input dataframe...\n")
    input_data = pd.read_pickle(FILEPATH + DATAFRAME_FILENAME)
    print ("Input Dataframe Pickle file loaded.\n")
    
    
    #Set the max number of features
    if(NUM_FEATURES > features.shape[1]):
        print("Invalid Value for NUM_FEATURES. Selecting All Features.")
        NUM_FEATURES = features.shape[1]
        
    
    #Reducing the number of features 
    test = SelectKBest(score_func=chi2, k=NUM_FEATURES)
    fit = test.fit(features, input_data['Helpful'])
    best_features = fit.transform(features)
    
    print("Splitting the data into training and testing sets....\n")
    X_trainSet, X_testSet, Y_train, Y_test = train_test_split(best_features.todense(),input_data['Helpful'], 
                                                        test_size=0.2, random_state=1000, stratify = input_data['Helpful'])
    
    #Appending Class Labels for training and Test Set
    X_trainSet = np.append(X_trainSet,Y_train.reshape(len(Y_train),1),1)
    X_testSet = np.append(X_testSet,Y_test.reshape(len(Y_test),1),1)
    
    #Removing Un-necessary Variables
    del input_data, Y_train
    
    print("Training and Test data set created successfully.\n")
    
    
    #Train and generating predictions with the KNN Classifier
    print ("Training and generating predictions with the KNN Classifier...\n")
    
    test_preds = []
    cnt = 0
    
    for X_test in X_testSet:
        
        if((cnt+1) % 1000 == 0):
            print (str(cnt+1) + " rows of test data processed.")
        
        neighbours = fun_get_kNeighbors(np.array(X_trainSet), np.array(X_test), K)
        result = fun_getPrediction(neighbours)
        test_preds.append(result)
        cnt = cnt+1

    print("Class Labels prediction completed.\n")
    
    #Confusion Matrix Values
    tn, fp, fn, tp = confusion_matrix(Y_test, test_preds).ravel()
    
    #Computing Precision, Recall, Accuracy and F1 Score Values
    if (tp > 0 or fp > 0 ):
        Precision = (float(tp)/(tp+fp)) * 100
    else :
        print ("Both true and false positives are zero.\n")
        Precision = 0
        
    if (tp > 0 or fn > 0):
        Recall = (float(tp)/(tp+fn)) * 100 
    else:
        print ("Both true positive and false negative are zero.\n")
        Recall = 0

    if (tp > 0 or fp > 0 or tn > 0 or fn > 0):
        Accuracy = (float(tp+tn)/(tp+fp+tn+fn))*100 
    else:
        print ("All values in Confusion Matrix is zero.\n")
        Accuracy = 0        
        
    if (Precision > 0 or Recall > 0):
        F1 = (2*Precision*Recall/(Precision+Recall))
    else:
        print ("Both Precision and Recall is zero.\n")
        F1 = 0
    
    #Computing Model Accuracy
    print ('KNN Model Accuracy: {0}%\n'.format(Accuracy))
    
    #Computing Model Precision
    print ('KNN Model Precision: {0}%\n'.format(Precision))
    
    #Computing Model Recall
    print ('KNN Model Recall: {0}%\n'.format(Recall))
    
    #Computing Model F1 Score
    print ('KNN Model F1 Score: {0}'.format(F1))

else:
    print("No Pre-processed Pickle Files found. Please execute the pre-processing script.\n")
    

#Elapsed Time
print("Elapsed Time --- %s Minutes ---" % round((time.time() - start_time)/float(60),2)) 
            
            
    