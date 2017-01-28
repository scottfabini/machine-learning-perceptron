from random import uniform
from Classifier import Classifier
import numpy as np
import math


class Layer:
    def __init__(self, layer_size, input_size, learning_rate, momentum):
        self.layer_size = layer_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.classifiers = [Classifier(input_size, learning_rate, momentum) for _ in range(layer_size)]

    def output_vector(self, image_input):
        return [self.classifiers[i].classifier_output(image_input) for i in range(self.layer_size)]

    def prediction(self, image_input):
        vector = self.output_vector(image_input)
        prediction = np.argmax(vector)
        return prediction.item()
