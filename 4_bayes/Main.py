#!/usr/bin/env python3

"""
Homework 3 for CS545 Machine Learning at Portland State University

Perform experiments on Support Vector Machines (SVM), using HP's Spambase data.
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


class Main:

    def main(self, argv):
        """
        Experiment 1:

        Split the data in 1/2, into training data and test data. Train the SVM using the training data.
        Then run the trained SVM on the test data. Report out on accuracy, precision, recall, and ROC curve.

        """

        # Shuffle and split the data into training and testing data.
        # X corresponds to raw data.
        # Y corresponds to label/classification.
        data = pd.read_csv('spambase/spambase.data', header=None, index_col=57)
        data = sklearn.utils.shuffle(data)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5)
       
        

        probability_class_is_zero = numpy.count_nonzero(X_train.index) / len(X_train.index)
        probability_class_is_one = (len(X_train.index) - numpy.count_nonzero(X_train.index)) / len(X_train.index)
        print("P(0): {0}, P(1): {1}".format(probability_class_is_zero * 100, probability_class_is_one * 100))

        X_train_classified_as_zero = X_train.iloc[numpy.where(X_train.index == 0)[0]]
        X_train_classified_as_one = X_train.iloc[numpy.where(X_train.index == 1)[0]]
         
        feature_means_given_zero = X_train_classified_as_zero.mean(axis=0)
        feature_means_given_one = X_train_classified_as_one.mean(axis=0)
        feature_std_given_zero = X_train_classified_as_zero.std(axis=0) 
        feature_std_given_one = X_train_classified_as_one.std(axis=0) 

        probability_x_given_class_is_zero =  [[1 / (std * math.sqrt(2 * math.pi)) * math.exp(-(x - mean)**2 / (2 * std **2)) \
            for mean, std, x in zip(feature_means_given_zero, feature_std_given_zero, x_rows) if mean != 0] for x_rows in X_train.values]
        probability_x_given_class_is_one =  [[1 / (std * math.sqrt(2 * math.pi)) * math.exp(-(x - mean)**2 / (2 * std **2)) \
            for mean, std, x in zip(feature_means_given_one, feature_std_given_one, x_rows) if mean != 0] for x_rows in X_train.values]
         
        
        class_of_x_given_zero = math.log(probability_class_is_zero) + numpy.array([numpy.sum([math.log(xi) for xi in x_rows if xi != 0]) for x_rows in probability_x_given_class_is_zero])
        class_of_x_given_one = math.log(probability_class_is_one) + numpy.array([numpy.sum([math.log(xi) for xi in x_rows if xi != 0]) for x_rows in probability_x_given_class_is_one])
        

        print("class | 0: {0}".format(class_of_x_given_zero))
        print("class | 1: {0}".format(class_of_x_given_one))

        class_prediction = [1 if a > b else 0 for a, b in zip(class_of_x_given_one, class_of_x_given_zero)]

        print(sklearn.metrics.classification_report(class_prediction, y_train))
        print(sklearn.metrics.confusion_matrix(class_prediction, y_train))
        
        return None

if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))