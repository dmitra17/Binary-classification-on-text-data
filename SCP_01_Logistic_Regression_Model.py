#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:19:22 2017

@authors: Group-8

Script Name: SCP_01_Logistic_Regression_Model.py

Script Description : This script loads the pre-processed files, split the data into 80-20% training-test set
                     and applies Logistic Regression model. It also displays the Accuracy, Precision, Recall and F1 Score.
                     
"""

#Import Libraries
from sklearn.cross_validation import train_test_split
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#FilePath
FILEPATH = "/Users/Avirup/Desktop/Fall, 2017/Introduction to Data Mining/Projects/Final Project/"

#Features file name
FEATURESET_FILENAME = "features_reviews_Clothing_Shoes_and_Jewelry.pickle"

#File containing the converted DataFrame
DATAFRAME_FILENAME = "reviews_Clothing_Shoes_and_Jewelry.pickle"


#Input Parameters

#Number of important features to be retained
NUM_FEATURES = 10
#Number of maximum iterations
NUM_ITER = 1000
#Learning rate for gradient computation
LEARNING_RATE = 5e-5



#Function to compute the sigmoid function
def fun_sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


#Function to compute the log likelihood values
def fun_log_likelihood(features, labels, weights):
    scores = np.dot(features, weights)
    ll = np.sum( labels*scores - np.log(1 + np.exp(scores)) )
    return ll

def fun_logistic_regression_train(features, labels, num_steps, learning_rate):
        
    weights = np.zeros(features.shape[1])
    
    for step in xrange(num_steps):
        
        #Calculate scores based on sigmoid function
        scores = np.dot(features, weights)
        predictions = fun_sigmoid(scores)

        #Weight updation with gradient ascent
        out_err_val = labels - predictions
        gradient = np.dot(features.T, out_err_val)
        weights += learning_rate * gradient
        
        # Print log-likelihood values of convergence 
        if step % 10 == 0:
            
            print ("Log Likelihood value for step : " + str(step) + " is " + str(fun_log_likelihood(features, labels, weights)) + "\n")
                
    return weights

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
    test = SelectKBest(score_func=chi2, k= NUM_FEATURES)
    fit = test.fit(features, input_data['Helpful'])
    best_features = fit.transform(features)
    
    print("Splitting the data into training and testing sets....\n")
    X_train, X_test, Y_train, Y_test = train_test_split(best_features.todense(),input_data['Helpful'], 
                                                        test_size=0.2, random_state=1000,stratify = input_data['Helpful'])
    print("Training and Test data set created successfully.\n")
    
    
    #Training the dataset using logistic regression
    print ("Training the dataset using logistic regression....\n")
    model_weights = fun_logistic_regression_train(np.array(X_train), Y_train, 
                                            num_steps = NUM_ITER, learning_rate = LEARNING_RATE)
    print("Logistic Regression Model Trained Successfully.\n")
    
    #Testing the dataset using the trained model weights
    print("Predicting Class Labels on Test Data...\n")
    final_scores = np.dot(np.array(X_test), model_weights)
    test_preds = np.round(fun_sigmoid(final_scores))
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
    print ('Logistic Regression Model Accuracy: {0}%\n'.format(Accuracy))
    
    #Computing Model Precision
    print ('Logistic Regression Model Precision: {0}%\n'.format(Precision))
    
    #Computing Model Recall
    print ('Logistic Regression Model Recall: {0}%\n'.format(Recall))
    
    #Computing Model F1 Score
    print ('Logistic Regression Model F1 Score: {0}'.format(F1))
    
    
else:
    print("No Pre-processed Pickle Files found. Please execute the pre-processing script.\n")
    
#Elapsed Time
print("\nElapsed Time --- %s Minutes ---" % round((time.time() - start_time)/float(60),2))  

