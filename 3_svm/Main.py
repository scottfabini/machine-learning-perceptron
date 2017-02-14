#!/usr/bin/env python3

"""
Homework 3 for CS545 Machine Learning at Portland State University

Perform experiments on Support Vector Machines (SVM), using HP's Spambase data.
https://archive.ics.uci.edu/ml/datasets/Spambase

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics
import sklearn.svm
import random
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
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values,
                                                                                    test_size=0.5)

        # Normalize the data by scaling to the training data
        scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

        print(np.shape(X_train_norm))
        print(np.shape(y_train))
        # Initialize SVM
        clf = sklearn.svm.SVC(kernel="linear")
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

        # Plot ROC Curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_score)

        plt.figure(1)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % 0.0)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        """
        Experiment 2:

        Select from the set of features with the highest contributions to the weight vector.
        Do this for the full range of features (m = 2 to 57) and graph the accuracy as the number
        of features taken into account increases
        """

        # Determine weight vector. Sort descending, by indices corresponding to max values.
        weight_vector = clf.coef_[0]
        weight_vector_max_sorted_by_index = weight_vector.argsort()[::-1]

        # Take an increasing number (m) of features into account, and plot their ROC Curve and accuracy for increasing m
        accuracies = []
        for m in range(2, 58):
            X_train_norm_subset = []
            y_train_subset = []
            # Take the m most important features and create a training subset from them
            for i in range(m):
                if i == 0:
                    X_train_norm_subset = X_train_norm[:, 0]
                    X_test_norm_subset = X_test_norm[:, 0]
                else:
                    # TODO: This subset generation via column_stack is very inefficient.
                    X_train_norm_subset = np.column_stack((X_train_norm_subset, X_train_norm[:, i]))
                    X_test_norm_subset = np.column_stack((X_test_norm_subset, X_test_norm[:, i]))
                y_train_subset = y_train


            # Reinitialize SVM
            clf = sklearn.svm.SVC(kernel="linear")

            # Guard against all-zero/all-one y class, which causes an exception in clf.fit()
            if not (all(v == 0 for v in y_train_subset) or all(v == 1 for v in y_train_subset)):
                clf.fit(X_train_norm_subset, y_train_subset)
                # Get scores/predictions from the trained SVM based for test inputs
                y_score = clf.decision_function(X_test_norm_subset)
                y_predict = clf.predict(X_test_norm_subset)
                #Determine accuracy
                accuracy = sklearn.metrics.accuracy_score(y_test[:], y_predict)
                accuracies.append(accuracy)

                # Plot ROC Curve
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test[:], y_score)
                plt.figure(2)
                plt.plot(fpr, tpr, color=cm.hot(m/100),
                         lw=lw, label='')

        # Display properties of Top 5 features with highest |w|
        print("Weight Vector Sorted By Highest-Weight Index: ")
        print(weight_vector_max_sorted_by_index)
        print("\n\n\n")

        # Finalize ROC Curve Plot
        plt.figure(2)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        # Plot accuracy vs. quantity of features (m)
        plt.figure(3)
        plt.plot(range(len(accuracies)), accuracies, color=cm.hot(m / 100), lw=lw, label='')
        plt.xlim([0.0, 57.0])
        plt.ylim([0.5, 1.01])
        plt.xlabel('m')
        plt.ylabel('Accuracy')
        plt.title('Accuracy as function of number of features')
        plt.legend(loc="lower right")

        """
        Experiment 3:

        Same as Experiment 2, but for each m, instead of selecting from the highest contributers to the
        weight vector, select m features at random from the complete set.
        """

        # Take an increasing number (m) of features into account, and plot their ROC Curve and accuracy for increasing m
        accuracies = []
        for m in range(2, 58):
            X_train_norm_subset = []
            y_train_subset = []
            # Take the m most important features and create a training subset from them
            for i in range(m):
                if i == 0:
                    X_train_norm_subset = X_train_norm[:, 0]
                    X_test_norm_subset = X_test_norm[:, 0]
                else:
                    i = random.randrange(57)
                    # TODO: This subset generation via column_stack is very inefficient.
                    X_train_norm_subset = np.column_stack((X_train_norm_subset, X_train_norm[:, i]))
                    X_test_norm_subset = np.column_stack((X_test_norm_subset, X_test_norm[:, i]))
                y_train_subset = y_train

            # Reinitialize SVM
            clf = sklearn.svm.SVC(kernel="linear")

            # Guard against empty (all-zero/all-one) y class, which causes an exception in clf.fit()
            if not (all(v == 0 for v in y_train_subset) or all(v == 1 for v in y_train_subset)):
                # Train the SVM
                clf.fit(X_train_norm_subset, y_train_subset)
                # Get scores/predictions from the trained SVM based for test inputs
                y_score = clf.decision_function(X_test_norm_subset)
                y_predict = clf.predict(X_test_norm_subset)

                # Determine accuracy
                accuracy = sklearn.metrics.accuracy_score(y_test[:], y_predict)
                accuracies.append(accuracy)

                # Plot ROC Curve
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test[:], y_score)
                plt.figure(4)
                plt.plot(fpr, tpr, color=cm.hot(m / 100),
                         lw=lw, label='')

        # Finalize ROC Curve Plot
        plt.figure(4)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        # Plot accuracy vs. quantity of features (m)
        plt.figure(5)
        plt.plot(range(len(accuracies)), accuracies, color=cm.hot(m / 100), lw=lw, label='')
        plt.xlim([0.0, 57.0])
        plt.ylim([0.5, 1.01])
        plt.xlabel('m')
        plt.ylabel('Accuracy')
        plt.title('Accuracy as function of number of features')
        plt.legend(loc="lower right")

        plt.show()

        return None

if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))