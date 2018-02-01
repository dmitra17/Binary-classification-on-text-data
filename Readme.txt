——————————————————————————————————————————————————————————————————
Original Input Files (Not included in the Zip file):-
——————————————————————————————————————————————————————————————————

	1. reviews_Clothing_Shoes_and_Jewelry.json.gz (This is the small dataset of 2.92 GB)
	2. reviews_Electronics.json.gz (This is the large dataset of 5.47 GB)

There are other input files that can be downloaded from the following website, apart from the files given above :
	
	http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/

There are certain .csv files as well in the above link but our code will only handle to files with .json.gz extension.

————————————————————————————————————————————————————————————————————
Summary of the files in Zip file (IDM_Final_Project_Files_Group8.zip)
————————————————————————————————————————————————————————————————————

Script Files :-
——————————————

	1. SCP_00_Pre_Processing_Reviews.py
	2. SCP_01_Logistic_Regression_Model.py
	3. SCP_02_Gaussian_Naive_Bayes_Model.py
	4. SCP_03_DecisionTree_Model_V1.py
	5. SCP_04_DecisionTree_Model_V2.py
	6. SCP_05_KNN_Model.py


Project Documentation :-
—————————————————————————

	1. IDM_Final_Project_Report_Group8.pdf
	2. IDM_Final_Project_Presentation_Group8.pptx
	3. Readme.txt


Python Libraries used in all the scripts:
—————————————————————————————————————————

1. numpy
2. os
3. six.moves
4. pandas
5. gzip
6. string
7. warnings
8. time
9. sklearn
10. collections
11. operator
12. nltk

Please check if all the above libraries are present by importing them in python. If not, then install them before execution of any of the scripts.  

—————————————————————————————————————————————————————————————
Script Execution Steps:
—————————————————————————————————————————————————————————————


1. Extract all the files from the zip file and place all of them in a folder.
2. Place the input data files in the same folder as created in Step 1. 
3. Open Spyder from Navigator Application. The scripts can also be executed from Command line using the following command:
 
	python <Script_Name>

The details of each script execution is given below :
—————————————————————————————————————————————————————

———————————————————————————————————
SCP_00_Pre_Processing_Reviews.py
———————————————————————————————————

The first script that needs to be executed before any other script is "SCP_00_Pre_Processing_Reviews.py"

1. Open the python file "SCP_00_Pre_Processing_Reviews.py" and set the following variables :

	a) FILEPATH --> Set the path of the input JSON Zip file to this variable.
	b) INPUT_JSON_FILENAME --> Set the filename of the input JSON Zip file.
	c) THRESHOLD --> Set the threshold value between 0 and 1 in order to generate the class labels.
	d) NGRAM_VAL --> Set an integer value between 1, 2 or 3 representing Unigram, Bigram or Trigram model.


2. Save the changes and execute the script either from Spyder or from Python Command-line. This script will generate two output pickle files. One with the prefix "reviews_" is having the input data converted from JSON to Dataframe. The second file with prefix "features_" contains the TF-IDF features generated from the reviewText column. 


————————————————————————————————————
SCP_01_Logistic_Regression_Model.py
————————————————————————————————————

After the pre-processing, apply logistic regression model using this script.

1. Open the python file "SCP_01_Logistic_Regression_Model.py" and set the following variables :

	a) FILEPATH --> Set the path of the pre-processed Dataframe and Features file to this variable. Please note that both files should be in the same directory.
	b) FEATURESET_FILENAME --> Set the feature set filename that was generated after pre-processing.
	c) DATAFRAME_FILENAME --> Set the dataframe filename that was generated after pre-processing.
	d) NUM_FEATURES --> Set an integer value of the number of features to be considered for Logistic Regression Model.
	e) NUM_ITER --> Set an integer value of the maximum number of iterations the model should re-compute the attribute weights.
	f) LEARNING_RATE --> Set a real value of the learning rate(alpha) which determines the gradient of the weights in each iteration. 


2. Save the changes and execute the script either from Spyder or from Python Command-line. The script will print the performance metrics of the model - Accuracy, Precision, Recall and F1-Score.


————————————————————————————————————
SCP_02_Gaussian_Naive_Bayes_Model.py
————————————————————————————————————

After the pre-processing, apply Gaussian Naive Bayes model using this script.

1. Open the python file "SCP_02_Gaussian_Naive_Bayes_Model.py" and set the following variables :

	a) FILEPATH --> Set the path of the pre-processed Dataframe and Features file to this variable. Please note that both files should be in the same directory.
	b) FEATURESET_FILENAME --> Set the feature set filename that was generated after pre-processing.
	c) DATAFRAME_FILENAME --> Set the dataframe filename that was generated after pre-processing.
	d) NUM_FEATURES --> Set an integer value of the number of features to be considered for Gaussian Naive Bayes Model. 

2. Save the changes and execute the script either from Spyder or from Python Command-line. The script will print the performance metrics of the model - Accuracy, Precision, Recall and F1-Score.

————————————————————————————————
SCP_03_DecisionTree_Model_V1.py
————————————————————————————————

After the pre-processing, apply Decision Tree model using this script. This is one of the implementations of the Decision Tree where each level has the same set of feature counts considered for split points computation.

1. Open the python file "SCP_03_DecisionTree_Model_V1.py" and set the following variables :

	a) FILEPATH --> Set the path of the pre-processed Dataframe and Features file to this variable. Please note that both files should be in the same directory.
	b) FEATURESET_FILENAME --> Set the feature set filename that was generated after pre-processing.
	c) DATAFRAME_FILENAME --> Set the dataframe filename that was generated after pre-processing.
	d) MODEL_FILENAME --> Set the name of the Decision Tree Model filename to be saved after training. It will be re-used on multiple executions of this script. In order to re-train the model, please 			      use a different model filename than the previous one.
	e) NUM_FEATURES --> Set an integer value of the number of features to be considered for Decision Tree Model.
	f) MAX_DEPTH --> Set an integer value of the maximum depth the tree should be constructed. It should be at-least 1.
	g) MIN_SIZE --> Set a integer value of the minimum number of rows each leaf node must contain in it. 


2. Save the changes and execute the script either from Spyder or from Python Command-line. The script will print the performance metrics of the model - Accuracy, Precision, Recall and F1-Score. It will also save the trained decision tree for re-executions of this script.

————————————————————————————————
SCP_04_DecisionTree_Model_V2.py
————————————————————————————————

After the pre-processing, apply Decision Tree model using this script. This is one of the implementations of the Decision Tree where each level has the one set of feature removed which was considered in each of the previous levels.

1. Open the python file "SCP_04_DecisionTree_Model_V1.py" and set the following variables :

	a) FILEPATH --> Set the path of the pre-processed Data frame and Features file to this variable. Please note that both files should be in the same directory.
	b) FEATURESET_FILENAME --> Set the feature set filename that was generated after pre-processing.
	c) DATAFRAME_FILENAME --> Set the data frame filename that was generated after pre-processing.
	d) MODEL_FILENAME --> Set the name of the Decision Tree Model filename to be saved after training. It will be re-used on multiple executions of this script. In order to re-train the model, please 			      use a different model filename than the previous one.
	e) NUM_FEATURES --> Set an integer value of the number of features to be considered for Decision Tree Model.
	f) MAX_DEPTH --> Set an integer value of the maximum depth the tree should be constructed. It should be at-least 1 and it should be less than the number of features for this implementation.
	g) MIN_SIZE --> Set a integer value of the minimum number of rows each leaf node must contain in it. 


2. Save the changes and execute the script either from Spyder or from Python Command-line. The script will print the performance metrics of the model - Accuracy, Precision, Recall and F1-Score. It will also save the trained decision tree for re-executions of this script.

————————————————————
SCP_05_KNN_Model.py
————————————————————

After the pre-processing, apply K-Nearest Neighbours model using this script.

1. Open the python file "SCP_05_KNN_Model.py" and set the following variables :

	a) FILEPATH --> Set the path of the pre-processed Dataframe and Features file to this variable. Please note that both files should be in the same directory.
	b) FEATURESET_FILENAME --> Set the feature set filename that was generated after pre-processing.
	c) DATAFRAME_FILENAME --> Set the dataframe filename that was generated after pre-processing.
	d) NUM_FEATURES --> Set an integer value of the number of features to be considered for K-Nearest Neighbours Model. 
	e) K --> Set an integer value for the number of nearest neighbours to be considered for computing the class label of the test data instance.

2. Save the changes and execute the script either from Spyder or from Python Command-line. The script will print the performance metrics of the model - Accuracy, Precision, Recall and F1-Score.


Please note that all the performance of the scripts is not optimum due to lack of usage of the built-in libraries as well as lack of hardware resources. It can be improved with better hardware and other optimisation techniques of parallelisation.



