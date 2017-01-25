from random import uniform


class Classifier:
    def __init__(self, input_size, learning_rate):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        for w in range(0, self.input_size + 1):
            weights.append(uniform(-0.5, 0.5))
        return weights

    def classifier_output(self, image_input):
        output = self.dot_product(image_input)
        return output

    def dot_product(self, image_input):
        return sum([i * w for i, w in zip(image_input, self.weights)])

    def adjust_weights(self, deltas):
        self.weights = [weight + delta for weight, delta in zip(self.weights, deltas)]
