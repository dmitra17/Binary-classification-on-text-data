#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:08:17 2017

@authors: Group-8

Script Name: SCP_00_Pre_Processing_Reviews.py

Script Description : This script pre-processes the input JSON files and generates two files.
                     One is the dataframe and the second is the text features file.

"""

#Import Libraries
import numpy as np
import os
from six.moves import cPickle as pickle
import pandas as pd
import gzip
import string
import warnings
import time
warnings.filterwarnings('ignore')

#Libraries for Text TF-IDF Computations
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

#Set stop words
stops = set(stopwords.words("english"))

#FilePath
FILEPATH = "/Users/Avirup/Desktop/Fall, 2017/Introduction to Data Mining/Projects/Final Project/"

#Input Data Set File
INPUT_JSON_FILENAME = "reviews_All_Credit_Cards.json.gz"


#Threshold value for generating class label of 0 or 1
THRESHOLD = 0.5

#N-Gram Value
NGRAM_VAL = 1


#File containing the converted DataFrame
DATAFRAME_FILENAME = (INPUT_JSON_FILENAME[:-8]+'.pickle')

#Store the features in a file
FEATURESET_FILENAME = ('features_'+INPUT_JSON_FILENAME[:-8]+'.pickle')

#Function to parse the input JSON file 
def fun_parse_json(path):
    lines = gzip.open(path, 'rb')
    for line in lines:
        yield eval(line)

#Function to convert the input JSON to DataFrame
def fun_generate_dataframe(path):
    i = 0
    data_dict = {}
    for d in fun_parse_json(path):
        
        #Filter those reviewers having more than 10 reviews
        
        if (d['helpful'][1] > 10) :
            data_dict[i] = d
            i += 1
    return pd.DataFrame.from_dict(data_dict, orient='index')



#Function to remove the punctuations from ReviewText Column
def fun_del_punctuations(text):
    return text.translate(None, string.punctuation.translate(None, '"')) 

#Function to tokenize the Review Text
def fun_generate_tokens(words):
    words = words.split()
    stems = []
    [stems.append(stemmer.stem(word)) for word in words]
    return stems

#Check the start time
start_time = time.time()

#Check if the feature set file is already present

if os.path.exists(FILEPATH + FEATURESET_FILENAME):
    print ('Pickled file containing features already present, loading...\n')
    features = pd.read_pickle(FILEPATH + FEATURESET_FILENAME)
    print ('Feature Set Pickle file loaded.\n')
    
else:
    
    #Read the input JSON file, convert it into the dataframe and store it in a pickle file
    
    #Loads dataframe pickle file if exists, else extracts and pickles it from input JSON file
    if os.path.exists(FILEPATH + DATAFRAME_FILENAME):
        print ('Pickled DataFrame file already present, loading input dataframe...\n')
        input_data = pd.read_pickle(FILEPATH + DATAFRAME_FILENAME)
        print ('Pickle DataFrame file loaded.\n')
    else:
        print ('Converting JSON to Dataframe....\n')
        input_data = fun_generate_dataframe(FILEPATH + INPUT_JSON_FILENAME)
         
        #Select the relevant feature columns from the input data frame
        print("Selecting the relevant columns for feature selection...\n")
        input_data = input_data.iloc[:, [6,5,3]]
        print("Feature Columns Selection completed.\n")
        
        #Split the numerator and denominator review counts from the helpful score array
        input_data['helpful_num'] = input_data['helpful'].apply(lambda x: x[0])
        input_data['helpful_denom'] = input_data['helpful'].apply(lambda x: x[1])
        
        #Delete un-needed 'helpful column from the dataframe
        del input_data['helpful']
        
        #Check if we have any null values
        print("Checking for nulls....\n")
        print (input_data.isnull().sum())
        
        #Remove Null Rows from dataframe
        input_data.dropna(how = 'any', inplace = True)
        
        #Generating the binary class labels of helpfulness with the threshold value
        print ("Generating the binary class labels of helpfulness...\n")
        input_data.loc[:, 'Helpful'] = np.where(input_data.loc[:, 'helpful_num'] / input_data.loc[:, 'helpful_denom'] > THRESHOLD, 1, 0)
        print ("Generating the binary class labels completed.\n")    
        
        # Generating the TF-IDF Features from the Review Text
        
        #Convert Review Text to Lowercase
        input_data.loc[:, 'reviewText'] = input_data['reviewText'].str.lower()
        
        # Remove the punctuations from Review Text
        input_data['reviewText']=input_data['reviewText'].apply( lambda x: fun_del_punctuations(x))
        
        #Create the pickle file with the formatted dataframe
        input_data.to_pickle(FILEPATH + DATAFRAME_FILENAME)
        print ('DataFrame file creation completed.\n')
    
    
    #Create a stemmer for Stemming words to its roots
    stemmer = SnowballStemmer("english")
    

    print("Generating the TF-IDF Features from the Review Text...\n")
        
    #Define the vectorizer
    vectorizer = TfidfVectorizer(tokenizer = fun_generate_tokens, stop_words = 'english', ngram_range=(1, NGRAM_VAL), min_df = 0.01)
        
    #Fit the vectorizers to the input dataframe
    features = vectorizer.fit_transform(input_data['reviewText'])
        
    #Write the feature set matrix to the pickle file
    with open(FILEPATH + FEATURESET_FILENAME, 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    

print("Pre-Processing Completed and Feature Set Generated Successfully.")   
#Elapsed Time
print("Elapsed Time --- %s Minutes ---" % round((time.time() - start_time)/float(60),2))     
    
