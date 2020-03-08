import numpy


class NeuralNetworkTester:
    """
    A class to test various Neural Networks performance
    for MNIST handwritten text
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 learning_rate, epoch, neural_network):
        """
        Construct Neural Network tester
        """
        # Hyperparameters
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.nn = neural_network(self.input_nodes, self.hidden_nodes,
                                 self.output_nodes, self.learning_rate)

        # Reset performance metrics
        self.correct_guesses = 0
        self.total_samples = 0

    def train_neural_network(self):
        """
        Train Neural Network with MNIST handwritten text dataset
        """
        training_data_vector = self._load_data_set("mnist_train.csv")

        for i in range(self.epoch):
            for sample in training_data_vector:
                # Separate label from image
                values = sample.split(',')

                label = int(values[0])
                # Image as a vector
                inputs = numpy.asfarray(values[1:])
                inputs = self._input_feature_scaling(inputs)

                # Create a vector of expected value with feature scaling
                targets = numpy.zeros(self.output_nodes) + 0.01
                targets[label] = 0.99

                # Train Neural Network
                self.nn.train(inputs, targets)

    def predict_neural_network(self):
        """
        Predict using Neural Network and calculate its performance
        """
        test_data_vector = self._load_data_set("mnist_test.csv")

        for sample in test_data_vector:
            # Separate label from image
            values = sample.split(',')

            expected_label = int(values[0])
            # Image as a vector
            inputs = numpy.asfarray(values[1:])
            inputs = self._input_feature_scaling(inputs)

            # Output from neural network
            outputs = self.nn.predict(inputs)
            actual_label = numpy.argmax(outputs)

            if actual_label == expected_label:
                self.correct_guesses += 1
            self.total_samples += 1

    def _load_data_set(self, path):
        data_file = open(path, "r")
        data_vector = data_file.readlines()
        data_file.close()
        return data_vector

    def _input_feature_scaling(self, vector):
        return (vector / 255.0 * 0.99) + 0.01

    def performance(self):
        print(f"{self.correct_guesses/self.total_samples * 100}%")
