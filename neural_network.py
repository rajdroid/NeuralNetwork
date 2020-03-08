class BaseNeuralNetwork:
    """
    The base class for Neural Networks.

    Every Neural Network should implement these methods according to
    Neural Network architecture.
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        Construct Neural Network
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

    def activation_function(self, x):
        """
        The activating function used by Neural Networks.

        The derived class should implement this according to
        Neural Network architecture (sigmoid, ReLU, etc.).
        """
        pass

    def train(self, input_vector, target_vector):
        """
        The Neural Network training function.

        The derived class should implement this according to
        Neural Network architecture.
        """
        pass

    def predict(self, input_vector):
        """
        The Neural Network prediction/inference function.

        The derived class should implement this according to
        Neural Network architecture.
        """
        pass
