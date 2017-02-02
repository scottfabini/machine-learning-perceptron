from random import uniform
import numpy as np
import math


class Classifier:
    def __init__(self, input_size, learning_rate, momentum):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = [uniform(-0.5, 0.5) for _ in range(0, self.input_size)]
        self.previous_weight_changes = [0 for _ in range(0, self.input_size)]

    def get_output(self, inputs_):
        z = self.dot_product(inputs_, self.weights)
        return self.activation_function(z)

    def dot_product(self, a, b):
        return np.dot(a, b)

    def activation_function(self, z):
        return 1 / (1 + math.e ** z)

    def update_weights(self, inputs_, error):
        np_inputs_ = np.array(inputs_)
        np_error = np.array(error)
        np_previous_weight_changes = np.array(self.previous_weight_changes)
        weight_changes = [self.learning_rate * np_error * np_inputs_[i]
                          + self.momentum * np_previous_weight_changes[i]
                          for i in range(0, self.input_size)]
        self.weights = [w + delta_w for w, delta_w in zip(self.weights, weight_changes)]
        self.previous_weight_changes = weight_changes

    def get_weights(self):
        return self.weights
