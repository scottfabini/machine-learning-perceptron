from random import uniform
import OutputClassifier
import numpy as np
import math


class OutputLayer():
    def __init__(self, layer_size, input_size, learning_rate, momentum, previous_layer):
        self.layer_size = layer_size
        self.classifiers = [OutputClassifier(input_size, learning_rate, momentum) for _ in range(layer_size)]
        self.weights_matrix = [classifier.get_weights() for classifier in self.classifiers]
        self.errors = [0 for _ in range(layer_size)]
        self.previous_layer = previous_layer

    def get_outputs_and_backpropagate(self, inputs_, target):
        outputs = self.get_outputs(inputs_)
        self.update_weights(inputs_, outputs, target)
        # should inputs_ here be the previous layers inputs?
        self.previous_layer.update_weights(inputs_, outputs, target)
        return outputs

    def get_outputs(self, inputs_):
        return [classifier.get_output(inputs_) for classifier in self.classifiers]

    def update_weights(self, inputs_, outputs, targets):
        for classifier in self.classifiers:
            classifier.update_weights(inputs_, self.get_errors(outputs, targets))

    def get_errors(self, outputs, targets):
        for idx in range(self.layer_size):
            self.errors[idx] = outputs[idx] * (1 - outputs[idx]) \
                               * (targets[idx] - outputs[idx])

    def get_weights_matrix(self):
        return self.weights_matrix




