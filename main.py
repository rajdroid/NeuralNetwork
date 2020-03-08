from two_layer_neural_network import TwoLayerNeuralNetwork
from neural_network_tester import NeuralNetworkTester

if __name__ == "__main__":
    neural_tester = NeuralNetworkTester(
        input_nodes=784,
        hidden_nodes=100,
        output_nodes=10,
        learning_rate=0.2,
        epoch=5,
        neural_network=TwoLayerNeuralNetwork
    )
    neural_tester.train_neural_network()
    neural_tester.predict_neural_network()
    neural_tester.performance()