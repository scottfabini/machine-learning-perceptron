from random import uniform
from OutputClassifier import OutputClassifier
import numpy as np
import math


class OutputLayer():
    def __init__(self, layer_size, input_size, learning_rate, momentum, previous_layer):
        self.layer_size = layer_size
        self.classifiers = [OutputClassifier(input_size, learning_rate, momentum) for _ in range(layer_size)]
        self.weights_matrix = [classifier.get_weights() for classifier in self.classifiers]
        self.errors = [0] * layer_size
        self.previous_layer = previous_layer

    def get_outputs_and_backpropagate(self, inputs_, targets):
        print('inputs: {0}'.format(inputs_))
        print('targets: {0}'.format(targets))
        outputs = self.get_outputs(inputs_)
        print('outputs: {0}'.format(outputs))
        errors = self.get_errors(outputs, targets)
        print('errors: {0}'.format(errors))
        #for classifier in self.classifiers:
        #    classifier.update_weights(inputs_, errors)
        # should inputs_ here be the previous layers inputs?
        #previous_errors = self.previous_layer.get_errors()
        #self.previous_layer.update_weights(inputs_, outputs, targets)
        return outputs

    def get_outputs(self, inputs_):
        return [classifier.get_output(inputs_) for classifier in self.classifiers]

    #def update_weights(self, inputs_, outputs, targets):
    #    for classifier in self.classifiers:
    #        classifier.update_weights(inputs_, self.get_errors(outputs, targets))

    def get_errors(self, outputs, targets):
        self.errors = [outputs[idx] * (1 - outputs[idx])
                       * (targets[idx] - outputs[idx])
                       for idx in range(self.layer_size)]
        return self.errors

    def get_weights_matrix(self):
        return self.weights_matrix




