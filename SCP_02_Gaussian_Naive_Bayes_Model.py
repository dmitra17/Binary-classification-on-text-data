#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:19:22 2017

@authors: Group-8

Script Name: SCP_02_Gaussian_Naive_Bayes_Model.py

Script Description : This script loads the pre-processed files, split the data into 80-20% training-test set
                     and applies Gaussian Naive Bayes model. It also displays the Accuracy, Precision, Recall and F1 Score.
    
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

#Number of important features to be retained
NUM_FEATURES = 10


#Function to seperate a dataset based on class labels
def fun_class_separation(feature_set, labels):
    
    #Use a dictionary with keys as labels and row instances as values
    separated_class_instances = {}

    #Class Label Zero
    separated_class_instances[0] = feature_set[np.where(labels==0)]
    
    #Class Label One
    separated_class_instances[1] = feature_set[np.where(labels==1)]
    
    return separated_class_instances


#Function to compute mean and standard deviation based on the classes
def fun_calc_mean_sd(feature_set):
    
    #Compute the mean and standard deviation of feature set    
    mean_stddev = zip(np.mean(feature_set, axis = 0),np.std(feature_set, axis = 0))
    
    return mean_stddev



#Function to separate dataset by class labels and compute mean and standard devitation of each attributes based on class
def fun_calc_mean_sd_ByClass(feature_set, labels):
    
    #Separate Class Intances by Class Labels
    separated_class_instances = fun_class_separation(feature_set, labels)
    mean_sd_by_class = {}
    
    #Compute Mean and Standard deviation based on class
    for classLabel, instances in separated_class_instances.iteritems():
        mean_sd_by_class[classLabel] = fun_calc_mean_sd(instances)
    return mean_sd_by_class


#Function to calculate probability density values
def fun_calc_Probability(x, mean, stdev):
    exponent = np.where(x > 0 , np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2)))) , -1)
    return np.where(exponent > 0 , (1 / (np.sqrt(2*np.pi) * stdev)) * exponent,  1)


#Function to calculate probability values by class
def fun_calc_ClassProbabilities(mean_sd_by_class, test_feature_vector):
    probabilities = {}
    for classLabel, mean_sd_class in mean_sd_by_class.iteritems():
        probabilities[classLabel] = 1
        for i in range(len(mean_sd_class)):
            mean, stdev = mean_sd_class[i]
            x = test_feature_vector[:,i]
            probabilities[classLabel] *= fun_calc_Probability(x, mean, stdev)
        
    return probabilities


#Function to predict class labels
def fun_predict_classLabels(probabilities):
    #Assign class probabilites of each class
    class_zero_probs = probabilities[0]
    class_one_probs = probabilities[1]
    #Predict labels
    preds_class_labels = np.where(class_one_probs > class_zero_probs,1,0)
    return preds_class_labels

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
                                                        test_size=0.2, random_state=1000, stratify = input_data['Helpful'])
    print("Training and Test data set created successfully.\n")
    
    #Train the Gaussian Naive Bayes Classifier
    print ("Training the Gaussian Naive Bayes Classifier...\n")
    model_mean_sd = fun_calc_mean_sd_ByClass(np.array(X_train), np.array(Y_train))
    print ("Gaussian Naive Bayes Classifier Trained Successfully.\n")
    
    #Testing the dataset using the trained model probabilites
    print("Predicting Class Labels on Test Data...\n")
    #Computing Probabilites
    test_probs = fun_calc_ClassProbabilities(model_mean_sd, np.array(X_test)) 
    #Computing Class Labels
    test_preds = fun_predict_classLabels(test_probs)
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
    print ('Gaussian Naive Bayes Model Accuracy: {0}%\n'.format(Accuracy))
    
    #Computing Model Precision
    print ('Gaussian Naive Bayes Model Precision: {0}%\n'.format(Precision))
    
    #Computing Model Recall
    print ('Gaussian Naive Bayes Model Recall: {0}%\n'.format(Recall))
    
    #Computing Model F1 Score
    print ('Gaussian Naive Bayes Model F1 Score: {0}'.format(F1))
    
else:
    print("No Pre-processed Pickle Files found. Please execute the pre-processing script.\n")
    
    
#Elapsed Time
print("\nElapsed Time --- %s Minutes ---" % round((time.time() - start_time)/float(60),2))      
    
    
