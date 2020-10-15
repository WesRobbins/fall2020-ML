from layer import *
from data_line import *

class MLP:

    def __init__(self, dataclass, classification_type):
        self.df = dataclass.df
        self.num_input_nodes = len(self.df.columns[:-1])
        self.num_hidden = 1
        self.n_outputs = len(self.df.Class.unique())
        input_row = DataLine(self.df.iloc[1]).feature_vector
        self.NN = self.initialize_network(self.num_input_nodes, self.num_hidden, 1,self.n_outputs)
        for layer in self.NN:
            print(layer)
        outputs = self.feed_forward(input_row)

        self.backpropagate([2, 4])

    def initialize_network(self, num_inputs, num_hidden, num_hidden_layers, num_outputs):
        """Initializes a neural network with an input layer, hidden layers, and output layers"""

        network = []
        for layer in range(num_hidden_layers):
            hidden_layer = Layer(1, num_inputs)
            network.append(hidden_layer)
        output_layer = Layer(num_outputs, num_hidden)
        network.append(output_layer)
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

        for i in reversed(range(len(self.NN))):
            current_layer = self.NN[i]
            errors = []
            if i == len(self.NN) - 1:
                for j in range(len(current_layer)):
                    cur_node = current_layer[j]
                    errors.append(expected[j] - cur_node.output)
            else:
                for j in range(len(current_layer)):
                    error = 0
                    for neuron in self.NN[i+1]:
                        error += neuron[j] * neuron.change
            for j in range(len(current_layer)):
                neuron = current_layer[j]
                neuron.change = errors[j] * neuron.derivative()




