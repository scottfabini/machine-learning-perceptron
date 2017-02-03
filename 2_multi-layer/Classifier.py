from random import uniform
import numpy as np
import math


class Classifier:
    def __init__(self, input_size, learning_rate, momentum):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = np.random.uniform(-0.05, 0.05, self.input_size)
        self.previous_weight_changes = np.zeros(input_size)

    def get_output(self, inputs_):
        # print(type(inputs_))
        # print(type(self.weights))
        z = np.dot(inputs_, self.weights)
        if z < -100: # protect against overflow
            z = -100
        if z > 100:
            z = 100
        return self.activation_function(z)

    def activation_function(self, z):
        return 1 / (1 + (math.exp(-z)))

    def update_weights(self, inputs_, error):
        weight_changes = (self.learning_rate * error) * inputs_ + (self.momentum * self.previous_weight_changes)
        self.weights +=  weight_changes
        self.previous_weight_changes = weight_changes
        return self.weights

    def get_weights(self):
        return self.weights

    def reset_previous_weight_changes(self):
        self.previous_weight_changes = np.zeros(self.input_size)
