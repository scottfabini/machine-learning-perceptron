from random import uniform
import HiddenClassifier
import numpy as np
import math


class HiddenLayer:
    def __init__(self, layer_size, input_size, learning_rate, momentum, previous_layer):
        self.layer_size = layer_size
        self.classifiers = [HiddenClassifier(input_size, learning_rate, momentum) for _ in range(layer_size)]
        self.errors = [0 for _ in range(layer_size)]
        self.previous_layer = previous_layer
        self.next_layer = None

    def get_outputs_and_backpropagate(self, inputs_, target):
        outputs = self.get_outputs(inputs_)
        self.update_weights(inputs_, outputs, target)
        if self.previous_layer is not None:
            #should these inputs be the previous layers inputs?
            self.previous_layer.update_weights(inputs_, outputs, target)
        return outputs

    def get_outputs(self, inputs_):
        return [classifier.get_output(inputs_) for classifier in self.classifiers]

    def update_weights(self, inputs_, outputs, targets):
        for classifier in self.classifiers:
            classifier.update_weights(inputs_, self.get_errors(outputs, targets))

    def get_errors(self, outputs):
        for idx in range(self.layer_size):
            self.errors[idx] = outputs[idx] * (1 - outputs[idx]) \
                               * np.dot(self.next_layer.get_weights_matrix(), self.next_layer.get_errors())

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer


