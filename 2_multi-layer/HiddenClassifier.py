from random import uniform
import numpy as np
import math
import Classifier


class HiddenClassifier(Classifier):
    def __init__(self, input_size, learning_rate, momentum):
        Classifier.__init__(input_size, learning_rate, momentum)

    def error(self, output, output_weights, output_errors):
        return output * (1 - output) * np.dot(output_weights, output_errors)

