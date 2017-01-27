from Classifier import Classifier
import numpy as np


class Perceptron:

    def __init__(self, input_size, learning_rate):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.perceptron_size = 10
        self.classifiers = [Classifier(input_size, learning_rate) for _ in range(self.perceptron_size)]

    def get_prediction(self, image):
        output = self.get_raw_output_vector(image)
        prediction = np.argmax(output)
        return prediction.item()

    def get_prediction_vector(self, image):
        prediction = self.get_prediction(image)
        output_vector = [0] * self.perceptron_size
        output_vector[prediction] = 1
        return output_vector

    def get_raw_output_vector(self, image):
        output = [0] * self.perceptron_size
        for i in range(self.perceptron_size):
            output[i] = self.classifiers[i].classifier_output(image)
        return output

    def exceeds_threshold(self, dotProduct):
        if dotProduct <= 0:
            return 0
        else:
            return 1

    def train(self, training_prediction, training_target, image):
        training_prediction_vector = self.number_to_vector(training_prediction)
        training_target_vector = self.number_to_vector(training_target)
        for i in range(self.perceptron_size):
            delta_ws = [self.learning_rate * (training_target_vector[i] - training_prediction_vector[i]) * x
                        for x in image]
            self.classifiers[i].adjust_weights(delta_ws)

    def number_to_vector(self, number):
        vector = [0] * self.perceptron_size
        vector[number] = 1
        return vector
