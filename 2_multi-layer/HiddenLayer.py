from random import uniform
from HiddenClassifier import HiddenClassifier
import numpy as np
import math


class HiddenLayer:
    def __init__(self, layer_size, input_size, learning_rate, momentum, previous_layer):
        self.layer_size = layer_size
        self.classifiers = [HiddenClassifier(input_size, learning_rate, momentum) for _ in range(layer_size)]
        self.errors = np.zeros(layer_size)
        self.previous_layer = previous_layer

    def get_outputs(self, inputs_):
        return [classifier.get_output(inputs_) for classifier in self.classifiers]

    def update_weights(self, inputs_, errors):
        for idx, classifier in enumerate(self.classifiers):
            classifier.update_weights(inputs_, errors[idx])

    def get_errors(self, outputs, weights_matrix, output_errors):
        for idx, classifier in enumerate(self.classifiers):
            self.errors[idx] = classifier.error(outputs[idx], weights_matrix[:, idx], output_errors)
        return np.array(self.errors)

    def reset_previous_weight_changes(self):
        for classifier in self.classifiers:
            classifier.reset_previous_weight_changes()

