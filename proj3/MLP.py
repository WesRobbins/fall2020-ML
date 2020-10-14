import math
from random import *
from layer import *

class MLP:

    def __init__(self, dataclass, classification_type):
        self.df = dataclass.df
        self.num_input_nodes = len(self.df.columns[:-1])
        self.num_hidden = 1
        self.num_outputs = 4
        self.NN = self.initialize_network(self.num_input_nodes, self.num_hidden, 1,self.num_outputs)

    def initialize_network(self, num_inputs, num_hidden, num_hidden_layers, num_outputs):
        """Initializes a neural network with an input layer, hidden layers, and output layers"""
        network = []
        for layer in range(num_hidden_layers):
            hidden_layer = Layer(1, num_inputs)
            network.append(hidden_layer)
        output_layer = Layer(num_outputs, num_hidden)
        network.append(output_layer)
        return network


    def feed_forward(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for node in layer:
                output = self.sum_weighted_inputs(node["weights"], inputs)


