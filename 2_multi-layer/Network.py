from Layer import Layer


class Network:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate, momentum):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer = Layer(hidden_layer_size, input_layer_size, learning_rate, momentum)
        self.output_layer = Layer(output_layer_size, hidden_layer_size, learning_rate, momentum)

    def vector(self, image):
        hidden_layer_result = self.hidden_layer.vector(image)
        return self.output_layer.vector(hidden_layer_result)
    
    def classification(self, image):
        hidden_layer_result = self.hidden_layer.vector(image)
        return self.output_layer.prediction(hidden_layer_result)


