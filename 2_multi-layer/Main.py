#!/usr/bin/env python3

from Network import Network
import sys
from ParseFile import Parser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import getopt
import numpy as np
import random
np.set_printoptions(threshold=np.nan)


class Main:

    def main(self, argv):

        # initialize command line parameter defaults
        hidden_layer_size = 20
        output_layer_size = 10
        momentum = 0.9
        learning_rate = 0.1
        epochs = 50
        max_train_inputs = 60000
        max_test_inputs = 10000
        accuracy_file = "accuracy.txt"
        confusion_file = "confusion.txt"
        randomize = False

        # process command line arguments
        try:
            opts, args = getopt.getopt(argv[1:], "n:m:l:h", ["max-train=", "max-test=", "accuracy-file=", "confusion-file="])
        except getopt.GetoptError:
            print('Main.py -n <hidden layer size> -o <output layer size> -m <momentum> -l <learning rate> -e <epochs>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('Main.py -n <hidden layer size> -o <output layer size> '
                      '-m <momentum> -l <learning_rate> -e <epochs>')
                sys.exit()
            elif opt == "-n":
                hidden_layer_size = int(arg)
            elif opt == "-o":
                output_layer_size = int(arg)
            elif opt == "-m":
                momentum = float(arg)
            elif opt == "-l":
                learning_rate = float(arg)
            elif opt == "-e":
                epochs = int(arg)
            elif opt == "--max-train":
                randomize = True
                max_train_inputs = int(arg)
            elif opt == "--max-test":
                max_test_inputs = int(arg)
            elif opt == "--accuracy-file":
                accuracy_file = './results/' + arg
            elif opt == "--confusion-file":
                confusion_file = './results/' + arg

        print('Configuration: \n'
              'Output Layers: {0}\n' 'Hidden Layers: {1}\n'
              'Epochs: {2}\n' 'Learning Rate: {3}\n'
              'Momentum: {4}\n'
              'Max Training Images: {5}\n' 'Max Test Images: {6}\n'
              'Accuracy File: {7}\n' 'Confusion File: {8}\n'
              .format(output_layer_size, hidden_layer_size, epochs, learning_rate, momentum,
                      max_train_inputs, max_test_inputs, accuracy_file, confusion_file))


        # parse input files
        parser = Parser()
        print('Begin Parsing Training Set...')
        training_images, training_targets = parser.parse_file('mnist_train.csv', 60000)

        # optionally pick randomized subset of training images
        if randomize:
            random_indexes = random.sample(range(60000), max_train_inputs)
            randomized_training_images = []
            randomized_training_targets = []
            for i in random_indexes:
                randomized_training_images.append(training_images[i])
                randomized_training_targets.append(training_targets[i])
            training_images = randomized_training_images
            training_targets = randomized_training_targets

        image_size = len(training_images[0])
        number_of_training_images = len(training_images)
        print('Parsing Training Set Complete.\n')

        print('Begin Parsing Test Set...')
        test_images, test_targets = parser.parse_file('mnist_test.csv', max_test_inputs)
        number_of_test_images = len(test_images)
        print('Parsing Test Set Complete.\n')

        # initialize network and per-epoch output variables
        network = Network(image_size, hidden_layer_size, output_layer_size, learning_rate, momentum)

        training_predictions = [0] * number_of_training_images
        training_prediction_vectors = [0] * number_of_training_images
        training_accuracies = []
        training_accuracy = 0

        test_predictions = [0] * number_of_test_images
        test_prediction_vectors = [0] * number_of_test_images
        test_accuracies = []

        # epoch 0 training setup
        training_prediction_vectors[0] = network.get_output(training_images[0])
        training_predictions[0] = self.vector_to_numeric(training_prediction_vectors[0])


        # begin learning
        print('Begin Learning With Learning Rate: {0}'.format(learning_rate))
        for epoch in range(epochs):

            # get test predictions
            for i in range(number_of_test_images):
                test_prediction_vectors[i] = network.get_output(np.array(test_images[i]))
                test_predictions[i] = self.vector_to_numeric(test_prediction_vectors[i])

            # accumulate results for this epoch
            previous_epoch_training_accuracy = training_accuracy
            training_accuracy = accuracy_score(training_targets, training_predictions) * 100
            training_accuracies.append(training_accuracy)
            test_accuracy = accuracy_score(test_targets, test_predictions) * 100
            test_accuracies.append(test_accuracy)

            # display intermediate results
            print('Epoch {0}'.format(epoch))
            print('Training Accuracy: {0:0.2f}%'.format(training_accuracy))
            print('Test Accuracy: {0:0.2f}%'.format(test_accuracy))
            print()

            # train based on predictions vs. targets
            for i in range(number_of_training_images):
                training_target_vector = self.numeric_to_vector(training_targets[i])
                training_prediction_vectors[i] = network.train(training_images[i], training_target_vector)
                training_predictions[i] = self.vector_to_numeric(training_prediction_vectors[i])

            # break out of epoch loop if training accuracy > 80% and delta is small
            if training_accuracy > 70 and abs(previous_epoch_training_accuracy - training_accuracy) <= 0.01:
                break

        # display final results
        print('Learning Rate: {0}'.format(learning_rate))
        print('Training Accuracies per Epoch and Final Confusion Matrix: ')
        print(str([value for value in training_accuracies]))
        print(str(confusion_matrix(training_targets, training_predictions)) + '\n')

        print('Test Accuracies per Epoch and Final Confusion Matrix: ')
        print(str([value for value in test_accuracies]))
        print(str(confusion_matrix(test_targets, test_predictions)) + '\n\n')

        # write accuracy results to files
        f = open(accuracy_file, 'w+')
        f.write(str([value for value in range(len(training_accuracies))]).strip('[]') + '\n')
        f.write(str([value for value in training_accuracies]).strip('[]') + '\n\n')
        f.write('Test Accuracies:\n')
        f.write(str([value for value in range(len(test_accuracies))]).strip('[]') + '\n')
        f.write(str([value for value in test_accuracies]).strip('[]') + '\n\n\n')
        f.close()

        # write confusion matrix results to files
        f = open(confusion_file, 'w+')
        f.write('Learning Rate: {0}\n'.format(learning_rate))
        f.write('Training Confusion Matrix: \n')
        f.write(str(confusion_matrix(training_targets, training_predictions)) + '\n\n')
        f.write('Test Confusion Matrix: \n')
        f.write(str(confusion_matrix(test_targets, test_predictions)) + '\n\n\n')
        f.close()

        return None

    # helper functions to convert from numeric value (e.g. 3) to array (.1,.1,.1,.9,.1,.1,.1,.1,.1) and back
    def vector_to_numeric(self, vector):
        return np.argmax(vector)

    def numeric_to_vector(self, numeric):
        vector = ([0.1]*10)
        vector[numeric] = 0.9
        return np.array(vector)

if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))


