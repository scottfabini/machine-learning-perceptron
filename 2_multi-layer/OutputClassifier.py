from random import uniform
import numpy as np
import math
from Classifier import Classifier


class OutputClassifier(Classifier):
    def __init__(self, input_size, learning_rate, momentum):
        Classifier.__init__(self, input_size, learning_rate, momentum)

    def error(self, output, target):
        return output * (1 - output) * (target - output)

