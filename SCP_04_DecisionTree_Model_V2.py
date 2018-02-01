#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:19:22 2017

@authors: Group-8

Script Name: SCP_04_DecisionTree_Model_V2.py

Script Description : This script loads the pre-processed files, split the data into 80-20% training-test set
                     and applies Decision Tree model with each level containing one feature less the previous level. 
                     It also displays the Accuracy, Precision, Recall and F1 Score.
                     
"""

#Import Libraries
from sklearn.cross_validation import train_test_split
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from six.moves import cPickle as pickle
import time

#FilePath
FILEPATH = "/Users/Avirup/Desktop/Fall, 2017/Introduction to Data Mining/Projects/Final Project/"

#Features file name
FEATURESET_FILENAME = "features_reviews_Clothing_Shoes_and_Jewelry.pickle"

#File containing the converted DataFrame
DATAFRAME_FILENAME = "reviews_Clothing_Shoes_and_Jewelry.pickle"

#File containing decision tree model
MODEL_FILENAME = "DT_Model_V2.pickle"

#Number of important features to be retained
NUM_FEATURES = 5

#Max Depth and Min Size of Tree
MAX_DEPTH = 3
MIN_SIZE = 2000

#Global Variables
global Feature_List
Feature_List = [999]
global LR_Flag
LR_Flag = 0

#Function to split rows of each attribute into two groups
def fun_split_groups(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Function to calculate the Gini index for a split dataset

def fun_calc_gini_index(groups, classes):

    gini = 0.0
    left_group, right_group = tuple(groups)
    left_group = np.array(left_group)
    right_group = np.array(right_group)
    left_size = float(len(left_group))
    right_size = float(len(right_group))
    n_instances = left_size + right_size
    
    # check to avoid divide by zero
    if left_size <> 0:
        score = 0.0
        score += np.sum(np.power(np.array(Counter(left_group[:,-1]).values()) / left_size,2))
        gini += (1.0 - score) * (left_size / n_instances)
    
    if right_size <> 0:
        score = 0.0
        score += np.sum(np.power(np.array(Counter(right_group[:,-1]).values()) / right_size,2))
        gini += (1.0 - score) * (right_size / n_instances)
    return gini


# Function to select the best split point for a dataset
def fun_calc_best_split(dataset):
    class_values = list(set(np.array(dataset)[:,-1]))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    
    for index in range(len(dataset[0])-1):
        
        if (index in Feature_List):
            continue
        
        row_num = 0
        for row in dataset:
            if((row_num+1) % 1000 == 0):
                print (str(row_num+1) +" rows processed for feature " + str(index+1))
            if(row[index] > 0):
                groups = fun_split_groups(index, row[index], dataset)
                gini = fun_calc_gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
            row_num = row_num + 1
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Function to create a terminal node value
def fun_gen_terminal(group):
    outcomes = list(np.array(group)[:,-1])
    return max(set(outcomes), key=outcomes.count)

# Function to create child splits for a node or generate a terminal node
def fun_gen_child_split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = fun_gen_terminal(left + right)
        del Feature_List[-1]
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = fun_gen_terminal(left), fun_gen_terminal(right)
        del Feature_List[-1]
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = fun_gen_terminal(left)
    else:
        node['left'] = fun_calc_best_split(left)
        Feature_List.append(node['left']['index'])
        fun_gen_child_split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = fun_gen_terminal(right)
        del Feature_List[-1]
    else:
        node['right'] = fun_calc_best_split(right)
        Feature_List.append(node['right']['index']) 
        fun_gen_child_split(node['right'], max_depth, min_size, depth+1)
 
# Function to build a decision tree
def fun_build_tree(train, max_depth, min_size):
    root = fun_calc_best_split(train)
    #Add New Feature to the list
    Feature_List.append(root['index'])
    print('Feature X%d < %.3f selected as Root Node ' % (root['index']+1, root['value']))
    fun_gen_child_split(root, max_depth, min_size, 1)
    
    return root

# Function to make a prediction with a decision tree
def fun_predict_labels(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return fun_predict_labels(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return fun_predict_labels(node['right'], row)
        else:
            return node['right']

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
        
    #Check the number of features and max depth
    if (NUM_FEATURES <= MAX_DEPTH):
        print ("Maximum Depth cannot be greater than Number of Features for this implementation. Selecting MAX_DEPTH value 1 less than NUM_FEATURES..\n")
        MAX_DEPTH = NUM_FEATURES - 1
    
    #Reducing the number of features 
    test = SelectKBest(score_func=chi2, k= NUM_FEATURES)
    fit = test.fit(features, input_data['Helpful'])
    best_features = fit.transform(features)
    
    print("Splitting the data into training and testing sets....\n")
    X_trainSet, X_testSet, Y_train, Y_test = train_test_split(best_features.todense(),input_data['Helpful'], 
                                                        test_size=0.2, random_state=1000, stratify = input_data['Helpful'])
    
    #Appending Class Labels for training and Test Set
    X_trainSet = np.array(np.append(X_trainSet,Y_train.reshape(len(Y_train),1),1))
    X_testSet = np.array(np.append(X_testSet,Y_test.reshape(len(Y_test),1),1))
    
    #Removing Un-necessary Variables
    del input_data, Y_train

    
    #Check if the decision tree model exists or not
    if os.path.exists(FILEPATH + MODEL_FILENAME):
        decisiontree_model = pd.read_pickle(FILEPATH + MODEL_FILENAME)
        print("Existing Decision Tree Model Loaded.\n")
    else:
        #Train the Decision Tree Classifier
        print ("Building the Decision Tree Model...\n")
        decisiontree_model = fun_build_tree(X_trainSet,MAX_DEPTH,MIN_SIZE)
        print ("Decision Tree Model built successfully.\n")
        
        #Write the decision tree model to the pickle file
        with open(FILEPATH + MODEL_FILENAME, 'wb') as handle:
            pickle.dump(decisiontree_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #Testing the dataset using the trained model probabilites
    print("Predicting Class Labels on Test Data...\n")
    #Computing Probabilites
    test_preds = [fun_predict_labels(decisiontree_model, test_inst) for test_inst in X_testSet]

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
    print ('Decision Tree Model Accuracy: {0}%\n'.format(Accuracy))
    
    #Computing Model Precision
    print ('Decision Tree Model Precision: {0}%\n'.format(Precision))
    
    #Computing Model Recall
    print ('Decision Tree Model Recall: {0}%\n'.format(Recall))
    
    #Computing Model F1 Score
    print ('Decision Tree Model F1 Score: {0}'.format(F1))

else:
    print("No Pre-processed Pickle Files found. Please execute the pre-processing script.\n")
    
#Elapsed Time
print("Elapsed Time --- %s Minutes ---" % round((time.time() - start_time)/float(60),2))