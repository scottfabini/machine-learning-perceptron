from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
import numpy as np

class Network:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate, momentum):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer = HiddenLayer(hidden_layer_size, input_layer_size, learning_rate, momentum, None)
        self.output_layer = OutputLayer(output_layer_size, hidden_layer_size, learning_rate, momentum, self.hidden_layer)

    def get_output(self, image):
        hidden_layer_results = self.hidden_layer.get_outputs(image)
        hidden_layer_results = np.insert(hidden_layer_results, 0, [1])
#        self.hidden_layer.get_weights()
#        print(hidden_layer_results)
        return self.output_layer.get_outputs(hidden_layer_results)

    def train(self, image, targets):
        hidden_layer_results = self.hidden_layer.get_outputs(image)
        hidden_layer_results = np.insert(hidden_layer_results, 0, [1])
        #print(hidden_layer_results)
        #print(self.output_layer.get_outputs_and_backpropagate(image, hidden_layer_results, targets))

        return self.output_layer.get_outputs_and_backpropagate(image, hidden_layer_results, targets)

    def reset_for_epoch(self):
        self.hidden_layer.reset_previous_weight_changes()
        self.output_layer.reset_previous_weight_changes()
