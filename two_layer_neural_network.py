import numpy

from neural_network import BaseNeuralNetwork


class TwoLayerNeuralNetwork (BaseNeuralNetwork):
    """
    A two layer Neural Network with sigmoid activation function.
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        Construct two layer Neural Network.
        """
        super().__init__(input_nodes, hidden_nodes,
                         output_nodes, learning_rate)

        # Assign random neural network weights (however not that random)
        self.Wi_h = numpy.random.normal(0.0, pow(hidden_nodes, -0.5),
                                        (hidden_nodes, input_nodes))
        self.Wh_o = numpy.random.normal(0.0, pow(output_nodes, -0.5),
                                        (output_nodes, hidden_nodes))

    def activation_function(self, x):
        return self._sigmoid(x)

    def _sigmoid(self, x):
        return 1/(1 + numpy.exp(-x))

    def train(self, input_vector, target_vector):
        """
        Train two layer Neural Network.
        """
        # Convert input and target vectors into two-dimensional array
        inputs = numpy.array(input_vector, ndmin=2).T
        targets = numpy.array(target_vector, ndmin=2).T

        hidden_outputs, final_outputs = self._feedforward(inputs)

        # Calculate error by taking difference of expected and actual
        output_errors = targets - final_outputs

        self._backpropagation(final_outputs, hidden_outputs,
                              inputs, output_errors)

    def predict(self, input_vector):
        """
        Predict value based on the input.
        """
        # Convert input into two-dimensional array
        inputs = numpy.array(input_vector, ndmin=2).T

        _, final_outputs = self._feedforward(inputs)

        return final_outputs

    def _feedforward(self, inputs):
        # Calculate hidden layer
        hidden_inputs = numpy.dot(self.Wi_h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate output layer
        final_inputs = numpy.dot(self.Wh_o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return hidden_outputs, final_outputs

    def _backpropagation(self, final_outputs, hidden_outputs,
                         inputs, output_errors):
        # Propagate error backward in proportion to weight of a neural link.
        hidden_errors = numpy.dot(self.Wh_o.T, output_errors)

        # Adjust weights of hidden to output layer according to error
        self.Wh_o += self.learning_rate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))

        # Adjust weights of input to hidden layer according to error
        self.Wi_h += self.learning_rate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))
