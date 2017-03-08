#!/usr/bin/env python3

"""
Homework 5 for CS545 Machine Learning at Portland State University

Perform experiments on K-Means Clustering, using Optdigits data.
http://web.cecs.pdx.edu/~mm/MachineLearningWinter2017/optdigits.zip
Each instance has 64 attributes, each of which can have value 0−16. Each instance also
has a label specifying which of 10 digit classes it belongs to.  

Example data:
00000000000000001110000000000000
00000000000000001110000000000000
00000000000000001110000000000000
00000000000000011100000000000000
00000000000000011100000000000000
00000000000000111100000000000000
00000000000000111100000000000000
00000000000000111100000000000000
00000000000001111100000000000000
00000000000001111100000000000000
00000000000001111000000000000000
00000000000111111000001111000000
00000000000111110000001111000000
00000000000111110000001111000000
00000000001111110000001111000000
00000000011111100000011111000000
00000000111111100000111111000000
00000001111111000000111111000000
00000011111111000000111110000000
00000011111111100000111110000000
00000011111111110001111110000000
00000001111111111111111100000000
00000001111111111111111100000000
00000000011111111111111100000000
00000000000000111111111000000000
00000000000000011111111000000000
00000000000000000111110000000000
00000000000000001111110000000000
00000000000000001111110000000000
00000000000000001111110000000000
00000000000000011111000000000000
00000000000000011111000000000000
 4

Corresponding data format (CSV): 0,0,0,1,11,7,0,0,0,0,2,13,10,16,4,0,0,0,13,4,1,16,0,0,0,6,14,8,12,16,7,0,0,0,8,8,15,10,2,0,0,0,0,1,12,1,0,0,0,0,0,4,16,0,0,0,0,0,0,3,15,0,0,0,4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import random
import math
import sys
import getopt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



np.set_printoptions(threshold=np.nan)


class Main:

    def main(self, argv):
        '''
        Experiment 1:

        Split the data in into training data and test data. Train the K-means clusterer using the training data.
        Then run the trained classifier on the test data. Report out on accuracy, precision, recall.

        '''
        print("\n\n*** Experiment 1 ***")
        #print(sys.argv)
        # process command line arguments
        # Get K-value, the size of the K-Cluster
        try:
            opts, args = getopt.getopt(argv[1:], "hK:")
        except getopt.GetoptError as err:
            print(str(err))
            print('Usage: Main.py -K <integer size of K-cluster, default K = 5>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('Main.py -K <integer size of K cluster, default K = 5>')
                sys.exit()
            elif opt == "-K":
                K = int(arg)
            else:
                K = 10
        K = 10
        # Shuffle and split the data into training and testing data.
        # X corresponds to raw data.
        # Y corresponds to label/classification.
        train_data = pd.read_csv('optdigits/optdigits.train', header=None)
        train_data = sklearn.utils.shuffle(train_data)
        y_train = train_data.iloc[:,-1]
        X_train = train_data
        X_train_values = train_data.drop(64, 1)


        test_data = pd.read_csv('optdigits/optdigits.test', header=None)
        test_data = sklearn.utils.shuffle(test_data)
        y_test = test_data.iloc[:,-1]
        #test_data = test_data.drop(64, 1)
        X_test = test_data

        # Run the K-means cluster algorithm.
        centroids = self.K_means_cluster(K, X_train_values)

        
        
        # Associate centroids with the most frequent class it contains.
        centroids = self.associate_most_frequent(K, centroids, X_train, y_train)

        

        # Calculate average mse:
        #print(self.mse(K, centroids, X_train))

        # Check the training data against the resultant clusters
        print("Training Set Results:\n")
        centroid_distances = self.distance(X_train, centroids)
        predictions = self.group_centroid_clusters(K, centroid_distances, X_train)
        print(predictions)
        print(y_train)

        #print("MSE: \n{0}".format(mean_squared_error(predictions, y_train)))
        
        print("Confusion Matrix: \n{0}".format(confusion_matrix(predictions, y_train)))

        # Run the test data against the resultant clusters
        print("Test Set Results:\n")
        centroid_distances = self.distance(X_test, centroids)
        predictions = self.group_centroid_clusters(K, centroid_distances, X_test)
        print(predictions)
        print(y_test)

        print("MSE: \n{0}".format(mean_squared_error(predictions, y_test)))
        print("Confusion Matrix: \n{0}".format(confusion_matrix(predictions, y_test)))
        print("Accuracy Score: \n{0}".format(accuracy_score(predictions, y_test)))

        # Plot the resulting centroids
        print("Centroids: \n{0}".format(centroids))

        self.plot_centroids(centroids)

        ### End main
    

    def mse(self, K, centroids, X_train):
        # Clusterize the data (again). TODO: Only clusterize once
        centroid_distances = self.distance(X_train, centroids)
        predictions = self.group_centroid_clusters(K, centroid_distances, X_train)
        clusters = self.cluster(K, predictions, X_train)  
        mse = [None] * K
        for i, df in enumerate(clusters):
            cluster_association = df.iloc[:,-1].value_counts().idxmax()

            mse[i] = self.mean_squared_error(df, centroids.iloc(cluster_association))
        return np.sum(mse)/K

    def mean_squared_error(self, df1, df2):
        np.sum(np.square(self.distance(df1, df2))) / len(df1.index)

    def K_means_cluster(self, K, X_train):
        '''
        K-means clustering algorithm.
        Args:
        K: Number of centroids to calculate.
        X_train (pd.DataFrame with N training data (rows), K (columns)): Training data.
        y_train (pd.DataFrame with N training data (rows))): Expected results for training data.

        Returns:
        K centroid values as a pd.DataFrame with shape (K, training data row length).
        '''
        
        centroids = X_train.head(K).set_index(np.arange(K))
        #centroids = pd.DataFrame(np.random.randint(0, 16, (K, 64)))
        previous_centroids = pd.DataFrame(np.random.randint(0, 16, (K, 64)))
        
        while (self.centroids_moving(centroids, previous_centroids) == False):
            previous_centroids = centroids
            centroid_distances = self.distance(X_train, centroids)
            grouped_centroid_clusters = self.group_centroid_clusters(K, centroid_distances, X_train)
            clusters = self.cluster(K, grouped_centroid_clusters, X_train)            
            centroids = self.move_centroids_to_means(K, clusters, centroids)
            #self.print_centroids(centroids)

        return centroids

    

    
    ''' Helper functions for K-means clustering '''
    def cluster(self, K, centroid_clusters, X_train):
        ''' 
        Create an array of K pandas Dataframes (i.e. a Panel),
        one for each cluster centroid. For each cluster centroid,
        append all of the training data that is closest to that centroid.

        '''
        clusters = [None] * K
        for i in np.arange(K):
            clusters[i] = pd.DataFrame()
        for index, row in X_train.iterrows():
            # Find the centroid cluster that this row is assigned to
            assigned_cluster = centroid_clusters.loc[index]
            # Append it into the DataFramefor that cluster.
            clusters[assigned_cluster] = clusters[assigned_cluster].append(row)
        return clusters        

    def move_centroids_to_means(self, K, clusters, centroids):
        '''
        Move the K centroids to the new center of the cluster data.
        '''
        mean_centroids = pd.DataFrame()
        for i, df in enumerate(clusters):
            new_centroid = df.mean(axis=0)
            
            if new_centroid.empty:
                mean_centroids = mean_centroids.append(centroids.iloc[i])
            else:
                mean_centroids = mean_centroids.append(new_centroid, ignore_index=True)
        return mean_centroids

    def group_centroid_clusters(self, K, distances, X_train):
        '''
        For each row index in the training set, set which cluster it belongs to,
        based on the minimum distance value in the distances data set.
        '''
        group = pd.Series()
        for index, row in distances.iterrows():
            #print("Row: {0}".format(row))
            centroid = row.argmin()
            group.set_value(index, centroid)
        return group

    
    def distance(self, X_train, centroids):
        '''
        Args:
        X_train (pd.DataFrame with N training data (rows), K (columns)): Training data.
        centroids (pd.DataFrame with N training data (rows), K (columns)): Current centroid locations.

        Returns:
        pd.DataFrame with N training data (rows), K (columns): The distances of the training data from the centroids
        '''
        distances = pd.DataFrame()
        for index, row in centroids.iterrows():
            distances[index] = np.sqrt(np.sum(np.square(X_train - row), axis=1))
        #distances = np.diag(distances)
        return distances
        

    def centroids_moving(self, centroids, new_centroids):
        '''
        Return true if all centroids moved a distance less than the threshold value.
        This decides whether the main K-means while loop stops.
        '''
        threshold = 1

        distances = pd.DataFrame()
        for index, row in centroids.iterrows():
            distances[index] = np.sqrt(np.sum(np.square(new_centroids - row), axis=1))

        distances = np.diag(distances)
        return np.all(distances < threshold)

        

    def associate_most_frequent(self, K, centroids, X_train, y_train):
        '''
        Associate the centroids with their most frequent classification.
        First, determine the mapping of current centroid to new centroid index value.
        Then reassign the index values for the centroid.
        '''
        centroid_distances = self.distance(X_train, centroids)
        predictions = self.group_centroid_clusters(K, centroid_distances, X_train)
        clusters = self.cluster(K, predictions, X_train)  

        centroids_map = pd.Series()
        for i, df in enumerate(clusters):
            # Find the most common item in the last column
            new_association = df.iloc[:,-1].value_counts().idxmax()
            centroids_map = centroids_map.append(pd.Series(int(new_association)))
        print("Centroids Map: \n{0}".format(centroids_map))
        reassociated_centroids = centroids.set_index(centroids_map).sort_index()
        return reassociated_centroids


    ''' Helper functions for plotting '''
    def print_centroids(self, centroids):
        print("Raw Centroids:\n")
        for index, row in centroids.iterrows():
            print(np.reshape(row, (8,8)))
            print("\n\n")
        centroids = self.normalize(centroids)
        print("Norm Centroids:\n")
        for index, row in centroids.iterrows():
            print(np.reshape(row, (8,8)))
            print("\n\n")

    def normalize(self, matrix):
        '''
        Normalize centroid matrix (which come in with values 0-16)
        to values 0-255, so they can be plotted in grayscale
        '''
        matrix = (matrix * 16).astype(int)
        return matrix

    def plot_centroids(self, centroids):
        '''
        Create K Grascale plots of centroids
        '''
        print("Centroids: \n{0}".format(centroids))
        #since there may be multiple indexes with the same value, brute-force creating multiple figures with i.
        i = 0
        for index, row in centroids.iterrows():

            row = self.normalize(row).tolist()
            #print("Pre-reshape: \n{0}".format(row))
            grayscale = np.reshape(row, (8, 8))
            print("Post-reshape: \n{0}\n\n".format(grayscale))

            plt.figure(i)
            print(grayscale)
            plt.style.use('grayscale') 
            plt.imshow(grayscale)
            i += 1
        
        plt.show()
       
if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))