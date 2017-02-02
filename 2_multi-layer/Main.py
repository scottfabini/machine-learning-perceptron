from Network import Network
import sys
from ParseFile import Parser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import getopt
import numpy as np


class Main:

    def main(self, argv):

        # process command line arguments
        try:
            opts, args = getopt.getopt(argv, "n:m:l:h", ["max-train=", "max-test="])
        except getopt.GetoptError:
            print('Main.py -n <hidden layer size> -o <output layer size> -m <momentum> -l <learning rate> -e <epochs>')
            sys.exit(2)
        hidden_layer_size = 20
        output_layer_size = 10
        momentum = 0.9
        learning_rate = 0.1
        epochs = 50
        max_train_inputs = 600
        max_test_inputs = 100
        for opt, arg in opts:
            if opt == '-h':
                print('Main.py -n <hidden layer size> -o <output layer size> -m <momentum> -l <learning_rate> -e <epochs>')
                sys.exit()
            elif opt in ("-n"):
                hidden_layer_size = arg
            elif opt in ("-o"):
                output_layer_size = arg
            elif opt in ("-m"):
                momentum = arg
            elif opt in ("-l"):
                learning_rate = arg
            elif opt in ("-e"):
                epochs = arg
            elif opt in ("--max-train"):
                max_train_inputs = arg
            elif opt in ("--max-test"):
                max_test_inputs = arg

        # parse input files
        parser = Parser()
        print('Begin Parsing Training Set...')
        training_images, training_targets = parser.parse_file('mnist_train.csv', max_train_inputs)
        image_size = len(training_images[0])
        number_of_training_images = len(training_images)
        print('Parsing Training Set Complete.\n')

        print('Begin Parsing Test Set...')
        test_images, test_targets = parser.parse_file('mnist_test.csv', max_test_inputs)
        number_of_test_images = len(test_images)
        print('Parsing Test Set Complete.\n')

        # initialize output files as blank
        f = open('accuracies.csv', 'w+')
        f.close()
        f = open('confusion.txt', 'w+')
        f.close()

        # initialize network and per-epoch output variables
        network = Network(image_size, hidden_layer_size, output_layer_size, learning_rate, momentum)

        training_predictions = [0] * number_of_training_images
        training_prediction_vectors = [0] * number_of_training_images
        training_accuracies = []
        training_accuracy = 0

        test_predictions = [0] * number_of_test_images
        test_prediction_vectors = [0] * number_of_test_images
        test_accuracies = []

        # begin learning
        print('Begin Learning With Learning Rate: {0}'.format(learning_rate))
        for epoch in range(epochs):
            # get training predictions
            for i in range(number_of_training_images):
                training_prediction_vectors[i] = network.get_output(training_images[i])
                training_predictions[i] = self.vector_to_numeric(training_prediction_vectors[i])
            previous_epoch_training_accuracy = training_accuracy
            training_accuracy = accuracy_score(training_targets, training_predictions) * 100
            training_accuracies.append(training_accuracy)

            # get test predictions
            for i in range(number_of_test_images):
                test_prediction_vectors[i] = network.get_output(test_images[i])
                test_predictions[i] = self.vector_to_numeric(test_prediction_vectors[i])
            test_accuracy = accuracy_score(test_targets, test_predictions) * 100
            test_accuracies.append(test_accuracy)

            # display intermediate results
            print('Epoch {0}'.format(epoch))
            print('Training Accuracy: {0:0.2f}%'.format(training_accuracy))
            print('Test Accuracy: {0:0.2f}%'.format(test_accuracy))
            print()

            # train based on predictions vs. targets
            for i in range(number_of_training_images):
                if training_predictions[i] != training_targets[i]:
                    training_target_vector = self.numeric_to_vector(training_targets[i])
                    network.train(training_images[i], training_target_vector)

            # break out of epoch loop if training accuracy > 80% and delta is small
            if abs(previous_epoch_training_accuracy - training_accuracy) <= 1.0:
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
        f = open('accuracies.csv'.format(learning_rate), 'a')
        f.write('Learning Rate: {0}\n'.format(learning_rate))
        f.write('Training Accuracies:\n')
        f.write(str([value for value in range(len(training_accuracies))]).strip('[]') + '\n')
        f.write(str([value for value in training_accuracies]).strip('[]') + '\n\n')
        f.write('Test Accuracies:\n')
        f.write(str([value for value in range(len(test_accuracies))]).strip('[]') + '\n')
        f.write(str([value for value in test_accuracies]).strip('[]') + '\n\n\n')
        f.close()

        # write confusion matrix results to files
        f = open('confusion.txt', 'a')
        f.write('Learning Rate: {0}\n'.format(learning_rate))
        f.write('Training Confusion Matrix: \n')
        f.write(str(confusion_matrix(training_targets, training_predictions)) + '\n\n')
        f.write('Test Confusion Matrix: \n')
        f.write(str(confusion_matrix(test_targets, test_predictions)) + '\n\n\n')
        f.close()

        return None

    def vector_to_numeric(self, vector):
        return np.argmax(vector)

    def numeric_to_vector(self, numeric):
        vector = ([0.1]*10)
        vector[numeric] = 0.9
        return vector

if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))


