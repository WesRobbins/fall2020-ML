from layer import *
from data_line import *

class MLP:

    def __init__(self, dataclass, classification_type):
        self.df = dataclass.df
        self.learning_rate = 1
        self.num_input_nodes = len(self.df.columns[:-1])
        self.num_hidden = 1
        self.n_outputs = len(self.df.Class.unique())
        input_row = DataLine(self.df.iloc[1]).feature_vector
        self.NN = self.initialize_network(self.num_input_nodes, self.num_hidden, 1,self.n_outputs)
        outputs = self.feed_forward(input_row)
        print(outputs)
        self.backpropagate([0, 1])

        self.update_node_weights(input_row)
        outputs = self.feed_forward(input_row)
        print(outputs)
        self.backpropagate([0, 1])

        self.update_node_weights(input_row)
        outputs = self.feed_forward(input_row)
        print(outputs)

    def initialize_network(self, num_inputs, num_hidden, num_hidden_layers, num_outputs):
        """Initializes a neural network with an input layer, hidden layers, and output layers"""

        network = []
        for layer in range(num_hidden_layers):
            hidden_layer = Layer(1, num_inputs)
            network.append(hidden_layer)
        self.output_layer = Layer(num_outputs, num_hidden)
        network.append(self.output_layer)
        return network


    def feed_forward(self, row):
        """Begins with inputs from the training set, runs it through the neural network and
        returns the output values"""

        inputs = row
        for layer in self.NN:
            new_inputs = []
            for node in layer:
                node.process_input(inputs)
                new_inputs.append(node.output)
            inputs = new_inputs
        return inputs

    def backpropagate(self, expected):
        """Backpropagates errors through neural network, assigning a delta weight value to each
        node. This delta weight value is the change that the node will make to its weight"""

        #Assigns delta values to each node in the output layer
        for i in range(len(self.output_layer)):
            node = self.output_layer[i]
            node.delta_weight = expected[i] - node.output * node.derivative()
            print(node.delta_weight)

        #Backpropagates errors through hidden layers
        for i in reversed(range(len(self.NN[:-1]))):
            errors = []
            layer = self.NN[i]
            for j in range(len(layer)):
                error = 0
                cur_node = layer[j]
                for node in self.NN[i+1]:
                    error += node.weights[j] * node.delta_weight
                cur_node.delta_weight = error * cur_node.derivative()

    def update_node_weights(self, inputs):

        for i in range(len(self.NN)):
            if i > 0:
                inputs = [node.output for node in self.NN[i-1]]
            for node in self.NN[i]:
                for j in range(len(inputs)):
                    node.weights[j] += self.learning_rate * node.delta_weight * inputs[j]
                node.weights[-1] += self.learning_rate * node.delta_weight




