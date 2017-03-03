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
import sklearn.linear_model
import sklearn.svm
import random
import math
import sys
numpy.set_printoptions(threshold=numpy.nan)


class Main:

    def main(self, argv):
        """
        Experiment 1:

        Split the data in into training data and test data. Train the Naive Gaussian Bayes classifier using the training data.
        Then run the trained classifier on the test data. Report out on accuracy, precision, recall.

        """
        print("\n\n*** Experiment 1 ***")
        # Shuffle and split the data into training and testing data.
        # X corresponds to raw data.
        # Y corresponds to label/classification.
        data = pd.read_csv('spambase/spambase.data', header=None, index_col=57)
        data = sklearn.utils.shuffle(data)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5)

        means_given_zero, std_given_zero, means_given_one, std_given_one = \
            self.determine_mean_std(X_train)

        # Training 
        class_prediction_X_train = \
            self.naive_bayes(X_train, means_given_zero, std_given_zero, means_given_one, std_given_one)
        
        # Training results
        print("Accuracy score: {0}".format(100 * sklearn.metrics.accuracy_score(class_prediction_X_train, y_train)))
        print(sklearn.metrics.classification_report(class_prediction_X_train, y_train))
        print(sklearn.metrics.confusion_matrix(class_prediction_X_train, y_train))

        # Testing 
        class_prediction_X_test = \
            self.naive_bayes(X_test, means_given_zero, std_given_zero, means_given_one, std_given_one)

        # Test results
        print("Accuracy score: {0}".format(100 * sklearn.metrics.accuracy_score(class_prediction_X_test, y_test)))
        print(sklearn.metrics.classification_report(class_prediction_X_test, y_test))
        print(sklearn.metrics.confusion_matrix(class_prediction_X_test, y_test))
        

        """
        Experiment 2:

        Train the Linear Regression classifier using the training data.
        Then run the trained classifier on the test data. Report out on accuracy, precision, recall.

        """
        print("\n\n*** Experiment 2 ***")
        # Normalize the data by scaling to the training data
        scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

        # Initialize Logistic Regression classifier
        clf = sklearn.linear_model.LogisticRegression()
        #clf = sklearn.svm.SVC(kernel="linear")
        # Train SVM using training data
        clf.fit(X_train_norm, y_train)

        # Classify the test data using the SVM.
        # Score is a weighted classification. Predict is 0/1 classification.
        y_score = clf.decision_function(X_test_norm)
        y_predict = clf.predict(X_test_norm)

        # Print results
        print("Test Set Accuracy Score:")
        print(sklearn.metrics.accuracy_score(y_test, y_predict))
        print("Test Set Precision Score:")
        print(sklearn.metrics.precision_score(y_test, y_predict))
        print("Test Set Recall Score:")
        print(sklearn.metrics.recall_score(y_test, y_predict))
        print("\n")

        print("Test Set Confusion Matrix:")
        print(sklearn.metrics.confusion_matrix(y_test, y_predict))
        print("\n")

        return None

    def determine_mean_std(self, X_train):
        # Split the data into those training data in class 1 and those in class 0
        X_train_classified_as_zero = X_train.iloc[numpy.where(X_train.index == 0)[0]]
        X_train_classified_as_one = X_train.iloc[numpy.where(X_train.index == 1)[0]]
         
        # Determine the means and the standard deviations of the training data in class 1 and class 0
        means_given_zero = X_train_classified_as_zero.mean(axis=0)
        std_given_zero = X_train_classified_as_zero.std(axis=0)

        std_given_zero = numpy.ma.masked_values(X_train_classified_as_zero.std(axis=0), 0).filled(.000001)
        means_given_one = X_train_classified_as_one.mean(axis=0)
        std_given_one = X_train_classified_as_one.std(axis=0)
        
        std_given_one = numpy.ma.masked_values(X_train_classified_as_one.std(axis=0), 0).filled(.000001)
        return means_given_zero, std_given_zero, means_given_one, std_given_one

    def naive_bayes(self, X_train, means_given_zero, std_given_zero, means_given_one, std_given_one):
        # Determine class probabilities P(0) and P(1)
        probability_class_is_one = numpy.count_nonzero(X_train.index) / len(X_train.index)
        probability_class_is_zero = (len(X_train.index) - numpy.count_nonzero(X_train.index)) / len(X_train.index)
        print("P(0): {0}, P(1): {1}".format(probability_class_is_zero * 100, probability_class_is_one * 100))
        # Deterimine P(xi | 0) and P(xi | 1) using Naive Bayes Gaussian formula
        probability_x_given_class_is_zero = 1 / (std_given_zero * numpy.sqrt(2 * math.pi)) \
                                                * numpy.exp(-(X_train.iloc[0] - means_given_zero)**2 / (2 * std_given_zero **2))

        probability_x_given_class_is_one = 1 / (std_given_one * numpy.sqrt(2 * math.pi)) \
                                               * numpy.exp(-(X_train.iloc[0] - means_given_one)**2 / (2 * std_given_one **2))
        for i in range(1, len(X_train) - 1):
            probability_xi_given_class_is_zero = 1 / (std_given_zero * numpy.sqrt(2 * math.pi)) \
                                                * numpy.exp(-(X_train.iloc[i] - means_given_zero)**2 / (2 * std_given_zero **2))

            probability_xi_given_class_is_one = 1 / (std_given_one * numpy.sqrt(2 * math.pi)) \
                                               * numpy.exp(-(X_train.iloc[i] - means_given_one)**2 / (2 * std_given_one **2))
            
            probability_x_given_class_is_zero = numpy.column_stack((probability_x_given_class_is_zero, probability_xi_given_class_is_zero))
            probability_x_given_class_is_one = numpy.column_stack((probability_x_given_class_is_one, probability_xi_given_class_is_one))
         
        '''
        # These lines can be uncommented to avoid divide by zero warnings. But accuracy goes down ~10%.
        probability_x_given_class_is_zero[probability_x_given_class_is_zero == 0] = 0.00000001
        probability_x_given_class_is_one[probability_x_given_class_is_one == 0] = 0.00000001
        '''

        # Determine class(X_train) = argmax {  log(P(0)) + sum(logs(xi|0))  }
        #                             {  log(P(1)) + sum(logs(xi|1))  }
        probability_class_of_x_is_zero = numpy.zeros(len(X_train))
        probability_class_of_x_is_one = numpy.zeros(len(X_train))
        for i in range(len(X_train) - 1):
            probability_class_of_x_is_zero[i] = numpy.log(probability_class_is_zero) + \
                                         numpy.sum(numpy.log(probability_x_given_class_is_zero[:,i]))
            probability_class_of_x_is_one[i] = numpy.log(probability_class_is_one) + \
                                        numpy.sum(numpy.log(probability_x_given_class_is_one[:,i])) 
        

        class_prediction = [1 if a >= b else 0 for a, b in \
                            zip(probability_class_of_x_is_one, probability_class_of_x_is_zero)]

        return class_prediction
        

if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))