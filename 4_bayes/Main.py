#!/usr/bin/env python3

"""
Homework 4 for CS545 Machine Learning at Portland State University

Perform experiments on Gaussian Bayes Classifiers, using HP's Spambase data.
https://archive.ics.uci.edu/ml/datasets/Spambase

"""

import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics
import sklearn.svm
import random
import math
import sys
#numpy.set_printoptions(threshold=numpy.nan)


class Main:

    def main(self, argv):
        """
        Experiment 1:

        Split the data in into training data and test data. Train the Naive Gaussian Bayes classifier using the training data.
        Then run the trained classifier on the test data. Report out on accuracy, precision, recall.

        """

        # Shuffle and split the data into training and testing data.
        # X corresponds to raw data.
        # Y corresponds to label/classification.
        data = pd.read_csv('spambase/spambase.data', header=None, index_col=57)
        data = sklearn.utils.shuffle(data)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5)

        # Determine class probabilities P(0) and P(1)
        probability_class_is_one = numpy.count_nonzero(X_train.index) / len(X_train.index)
        probability_class_is_zero = (len(X_train.index) - numpy.count_nonzero(X_train.index)) / len(X_train.index)
        print("P(0): {0}, P(1): {1}".format(probability_class_is_zero * 100, probability_class_is_one * 100))

        
        # Split the data into those training data in class 1 and those in class 0
        X_train_classified_as_zero = X_train.iloc[numpy.where(X_train.index == 0)[0]]
        X_train_classified_as_one = X_train.iloc[numpy.where(X_train.index == 1)[0]]
         
        # Determine the means and the standard deviations of the training data in class 1 and class 0
        feature_means_given_zero = X_train_classified_as_zero.mean(axis=0)
        feature_std_given_zero = X_train_classified_as_zero.std(axis=0) 
        feature_means_given_one = X_train_classified_as_one.mean(axis=0)
        feature_std_given_one = X_train_classified_as_one.std(axis=0) 

        # Deterimine P(xi | 0) and P(xi | 1) using Naive Bayes Gaussian formula
        probability_x_given_class_is_zero = 1 / (feature_std_given_zero * numpy.sqrt(2 * math.pi)) \
                                                * numpy.exp(-(X_train.iloc[0] - feature_means_given_zero)**2 / (2 * feature_std_given_zero **2))

        probability_x_given_class_is_one = 1 / (feature_std_given_one * numpy.sqrt(2 * math.pi)) \
                                               * numpy.exp(-(X_train.iloc[0] - feature_means_given_one)**2 / (2 * feature_std_given_one **2))
        for i in range(2300):    
            probability_xi_given_class_is_zero = 1 / (feature_std_given_zero * numpy.sqrt(2 * math.pi)) \
                                                * numpy.exp(-(X_train.iloc[i] - feature_means_given_zero)**2 / (2 * feature_std_given_zero **2))

            probability_xi_given_class_is_one = 1 / (feature_std_given_one * numpy.sqrt(2 * math.pi)) \
                                               * numpy.exp(-(X_train.iloc[i] - feature_means_given_one)**2 / (2 * feature_std_given_one **2))
            
            probability_x_given_class_is_zero = numpy.column_stack((probability_x_given_class_is_zero, probability_xi_given_class_is_zero))
            probability_x_given_class_is_one = numpy.column_stack((probability_x_given_class_is_one, probability_xi_given_class_is_one))
        
        #return probability_x_given_class_is_zero, probability_x_given_class_is_one
            
        # Determine class(x) = argmax {  log(P(0)) + sum(logs(xi|0))  }
        #                             {  log(P(1)) + sum(logs(xi|1))  }
        probability_class_of_x_is_zero = numpy.zeros(2300)
        probability_class_of_x_is_one = numpy.zeros(2300)
        for i in range(2300):
            probability_class_of_x_is_zero[i] = numpy.log(probability_class_is_zero) + \
                                         numpy.sum(numpy.ma.log(probability_x_given_class_is_zero[:,i]))
            probability_class_of_x_is_one[i] = numpy.log(probability_class_is_one) + \
                                        numpy.sum(numpy.ma.log(probability_x_given_class_is_one[:,i])) 
        
        class_prediction_X_train = [1 if a > b else 0 for a, b in \
                                   zip(probability_class_of_x_is_one, probability_class_of_x_is_zero)]

        # Display results
        
        

        #print("class | 0: {0}".format(numpy.shape(probability_class_of_x_is_zero)))
        #print("class | 1: {0}".format(numpy.shape(probability_class_of_x_is_one)))
        #print("class prediction: {0}".format(numpy.shape(class_prediction_X_train)))
        #print("class prediction: {0}".format(numpy.shape(y_train)))

        print("Accuracy score: {0}".format(100 * sklearn.metrics.accuracy_score(class_prediction_X_train, y_train)))
        print(sklearn.metrics.classification_report(class_prediction_X_train, y_train))
        print(sklearn.metrics.confusion_matrix(class_prediction_X_train, y_train))
        
        '''
        print(numpy.shape(feature_means_given_one))
        print(numpy.shape(feature_means_given_zero))
        print(numpy.shape(X_train))
        '''
        '''
        print("*******")
        print(math.log(probability_class_is_zero))
        print(probability_class_of_x_is_zero)
        print("*******")
        print(math.log(probability_class_is_one))
        print(probability_class_of_x_is_one)
        print("*******")
        #print("\n\n\n\n")
        #print(probability_x_given_class_is_zero)
        #print("\n\n\n\n")  
        #print(probability_x_given_class_is_one)
        #print(X_train_classified_as_zero)
        #print("*******")
        #print(X_train_classified_as_one)
        '''


        '''
        probability_class_of_x_is_zero = math.log(probability_class_is_zero) + \
            numpy.array([numpy.sum([numpy.log(xi) for xi in x_rows if xi != 0]) 
                                                 for x_rows in probability_x_given_class_is_zero])
        probability_class_of_x_is_one = math.log(probability_class_is_one) + \
            numpy.array([numpy.sum([numpy.log(xi) for xi in x_rows if xi != 0]) 
                                                 for x_rows in probability_x_given_class_is_one])
        '''
        return None

if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))