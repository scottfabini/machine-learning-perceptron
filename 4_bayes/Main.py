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

        X_train_classified_as_zero = numpy.transpose(X_train.iloc[numpy.where(X_train.index == 0)[0]])
        X_train_classified_as_one = numpy.transpose(X_train.iloc[numpy.where(X_train.index == 1)[0]])
  
        #marginal_probability_of_data 
                

        X_means_given_zero = numpy.mean(X_train_classified_as_zero, axis=1)
        X_means_given_one = numpy.mean(X_train_classified_as_one, axis=1)
        X_std_given_zero = numpy.std(X_train_classified_as_zero, axis=1) 
        X_std_given_one = numpy.std(X_train_classified_as_one, axis=1) 
        print("\n\nmean(X|1):\n{0}, \nmean(X|0):\n{1}".format(X_means_given_one, X_means_given_one))
        print("\n\nstd(X|1):\n{0}, \nstd(X|0):\n{1}".format(X_std_given_one, X_std_given_one))
        print("\n\nX classified 0:\n{0}".format(X_train_classified_as_zero))
        print("\n\nX classified 1:\n{0}".format(X_train_classified_as_one))
        
        '''
        print("\n\nX classified 0:\n{0}".format(numpy.shape(numpy.transpose(X_train_classified_as_zero))))
        print("\n\nX classified 1:\n{0}".format(numpy.shape(numpy.transpose(X_train_classified_as_one))))
        print("\n\nmean(X|1):\n{0}, \nmean(X|0):\n{1}".format(numpy.shape(X_means_given_one), numpy.shape(X_means_given_one)))
        print("\n\nstd(X|1):\n{0}, \nstd(X|0):\n{1}".format(numpy.shape(X_std_given_one), numpy.shape(X_std_given_one)))

        print("\n\nzip(X|1):\n{0}".format(list(zip(X_means_given_one, X_std_given_one, numpy.transpose(X_train_classified_as_one)))))
        print("\n\nzip(X|0):\n{0}".format(list(zip(X_means_given_zero, X_std_given_zero, numpy.transpose(X_train_classified_as_zero)))))
        print("\n\nzip(X|1):\n{0}".format(numpy.shape(list(zip(X_means_given_one, X_std_given_one, numpy.transpose(X_train_classified_as_one))))))
        print("\n\nzip(X|0):\n{0}".format(numpy.shape(list(zip(X_means_given_zero, X_std_given_zero, numpy.transpose(X_train_classified_as_zero))))))
        '''
        probability_x_given_class_is_zero =  [1 / (std * math.sqrt(2 * math.pi)) * math.exp(-(x - mean)**2 / (2 * std **2)) \
            for mean, std, x in zip(X_means_given_zero, X_std_given_zero, X_train_classified_as_zero) if mean != 0]
        probability_x_given_class_is_one =  [1 / (std * math.sqrt(2 * math.pi)) * math.exp(-(x - mean)**2 / (2 * std **2)) \
            for mean, std, x in zip(X_means_given_one, X_std_given_one, X_train_classified_as_one) if mean != 0]
         
        probability_of_feature = [numpy.sum(x * probability_class_is_one) for x in probability_x_given_class_is_one]
        print("P(x|c=0): {0}\n\n Shape: {1}\n\n".format(probability_x_given_class_is_zero, numpy.shape(probability_x_given_class_is_zero)))
        print("P(x|c=1): {0}\n\n, Shape: {1}\n\n".format(probability_x_given_class_is_one, numpy.shape(probability_x_given_class_is_one)))
        print("Probability of feature: {0}".format(probability_of_feature))
        
        class_of_x_given_zero = math.log(probability_class_is_zero) + numpy.sum([math.log(xi_given_class) for xi_given_class in probability_x_given_class_is_zero if xi_given_class != 0])
        class_of_x_given_one = math.log(probability_class_is_one) + numpy.sum([math.log(xi_given_class) for xi_given_class in probability_x_given_class_is_one if xi_given_class != 0])
        
        print("class | 0: {0}".format(class_of_x_given_zero))

        print("class | 1: {0}".format(class_of_x_given_one))
        '''
        [ numpy.argmax( math.log())for i in range(57)]
        # Normalize the data by scaling to the training data
        #scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train_norm = X_train
        X_test_norm = X_test

        #print(numpy.shape(X_train_norm))
        #print(numpy.shape(y_train))
        # Initialize SVM
        #clf = sklearn.svm.SVC(kernel="linear")
        # Train SVM using training data
        clf.fit(X_train_norm, y_train)

        # Classify the test data using the SVM.
        # Score is a weighted classification. Predict is 0/1 classification.
        y_score = clf.decision_function(X_test_norm)
        y_predict = clf.predict(X_test_norm)

        # Print results
        print("Accuracy Score:")
        print(sklearn.metrics.accuracy_score(y_test, y_predict))
        print("Precision Score:")
        print(sklearn.metrics.precision_score(y_test, y_predict))
        print("Recall Score:")
        print(sklearn.metrics.recall_score(y_test, y_predict))
        print("\n\n\n")

        print("Confusion Matrix:")
        print(sklearn.metrics.confusion_matrix(y_test, y_predict))
        print("\n\n\n")

        '''
        return None

if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))