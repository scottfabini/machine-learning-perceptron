from Layer import Layer


class Network:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate, momentum):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer = Layer(hidden_layer_size, input_layer_size, learning_rate, momentum)
        self.output_layer = Layer(output_layer_size, hidden_layer_size, learning_rate, momentum)

    def output_vector(self, image):
        intermediate_result = self.hidden_layer.output_vector(image)
        return self.output_layer.output_vector(intermediate_result)

    def prediction(self, image):
        intermediate_result = self.hidden_layer.output_vector(image)
        return self.output_layer.prediction(intermediate_result)
