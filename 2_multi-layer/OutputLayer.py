from random import uniform
from OutputClassifier import OutputClassifier
import numpy as np
import math


class OutputLayer:
    def __init__(self, layer_size, input_size, learning_rate, momentum, previous_layer):
        self.layer_size = layer_size
        self.classifiers = list([OutputClassifier(input_size + 1, learning_rate, momentum) for _ in range(layer_size)])
        self.weights_matrix = np.array(list([classifier.get_weights() for classifier in self.classifiers]))
        self.errors = [0] * layer_size
        self.previous_layer = previous_layer

    def get_outputs(self, inputs_):
        return [classifier.get_output(inputs_) for classifier in self.classifiers]

    def get_outputs_and_backpropagate(self, hidden_layer_inputs, output_layer_inputs, output_layer_targets):
        outputs = self.get_outputs(output_layer_inputs)
        # backpropagation within output layer
        output_layer_errors = self.get_errors(outputs, output_layer_targets)
        self.weights_matrix = np.array([classifier.update_weights(output_layer_inputs, output_layer_errors[idx]) for idx, classifier in enumerate(self.classifiers)])

        # backpropagation to hidden layer
        hidden_layer_outputs = output_layer_inputs
        hidden_layer_errors = self.previous_layer.get_errors(hidden_layer_outputs, self.weights_matrix, output_layer_errors)

        self.previous_layer.update_weights(hidden_layer_inputs, hidden_layer_errors)
        return outputs


    #def update_weights(self, inputs_, outputs, targets):
    #    for classifier in self.classifiers:
    #        classifier.update_weights(inputs_, self.get_errors(outputs, targets))

    def get_errors(self, outputs, targets):
        #print('outputs: {0}'.format(outputs))
        #print('targets: {0}'.format(targets))
        return [classifier.error(outputs[idx], targets[idx]) for idx, classifier in enumerate(self.classifiers)]

    def get_weights_matrix(self):
        return self.weights_matrix

    def reset_previous_weight_changes(self):
        for classifier in self.classifiers:
            classifier.reset_previous_weight_changes()

    '''
            sum_vector = []
            hidden_layer_outputs = output_layer_inputs
            scalar_terms = [hidden_output * (1 - hidden_output) for hidden_output in hidden_layer_outputs]
            #scalar_terms = [0] * len(hidden_layer_outputs)
            sum_terms = [0] * len(hidden_layer_outputs)
            hidden_layer_errors = [0] * len(hidden_layer_outputs)
            for j, hidden_layer_output in enumerate(hidden_layer_outputs):
                sum_terms[j] = np.dot(output_layer_errors, [weights[j] for weights in self.weights_matrix])
                    #sum([output_layer_error * self.classifiers for output_layer_error in output_layer_errors])
                hidden_layer_errors[j] = scalar_terms[j] * sum_terms[j]
            #print('sum_terms: {0}'.format(sum_terms))
            #print('hidden_layer_outputs: {0}'.format(hidden_layer_outputs))
            #print('hidden_layer_errors: {0}'.format(hidden_layer_errors))
            self.previous_layer.update_weights(hidden_layer_inputs, hidden_layer_errors)


            # should inputs_ here be the previous layers inputs?
            #previous_errors = self.previous_layer.get_errors()
            #self.previous_layer.update_weights(hidden_layer_inputs, previous_errors)
            return outputs
            '''
