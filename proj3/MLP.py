from layer import *
from data_line import *

class MLP:

    def __init__(self, dataclass, classification_type):
        self.df = dataclass.df  #Initializes the dataset
        self.initialize_parameters()    #Sets initial parameters
        #Grabs a row from the dataset for testing
        input_row = DataLine(self.df.iloc[1]).feature_vector
        #Initializes layers and nodes in network
        self.NN = self.initialize_network()
        #First pass of inputs through network
        outputs = self.feed_forward(input_row)
        print(f"Expected: [0, 1]")
        print(f"Actual: {outputs}")
        self.backpropagate([0, 1])  #Backpropagates weights with expected values
        self.update_node_weights(input_row)
        #Second pass of inputs through network
        outputs = self.feed_forward(input_row)
        print(f"Expected: [0, 1]")
        print(f"Actual: {outputs}")
        self.backpropagate([0, 1])
        self.update_node_weights(input_row)
        #Third pass of inputs through network
        outputs = self.feed_forward(input_row)
        print(f"Expected: [0, 1]")
        print(f"Actual: {outputs}")

    def initialize_parameters(self):
        """Initializes a set of parameters for the neural network"""

        self.n_inputs = len(self.df.columns[:-1])
        self.n_hidden_per_layer = 1
        self.n_hidden = 1
        self.n_outputs = len(self.df.Class.unique())
        self.learning_rate = 1

    def initialize_network(self):
        """Initializes a neural network with an input layer, hidden layers, and output layers"""

        network = []
        #Creates hidden layers
        for layer in range(self.n_hidden):
            hidden_layer = Layer(1, self.n_inputs)
            network.append(hidden_layer)
        self.output_layer = Layer(self.n_outputs, self.n_hidden_per_layer)
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
            #print(node.delta_weight)

        #Backpropagates errors through hidden layers
        for i in reversed(range(len(self.NN[:-1]))):
            errors = []
            layer = self.NN[i]
            #Iterates through each node in a layer
            for j in range(len(layer)):
                error = 0
                cur_node = layer[j]
                #Iterates through each node in the next layer up
                for node in self.NN[i+1]:
                    error += node.weights[j] * node.delta_weight
                cur_node.delta_weight = error * cur_node.derivative()

    def update_node_weights(self, inputs):
        """Goes through and updates all the weights utilizing input values, node weights,
        and the learning rate"""

        #Iterates through each layer
        for i in range(len(self.NN)):
            #If the layer is not the first layer, use previous layers outputs as inputs
            if i > 0:
                inputs = [node.output for node in self.NN[i-1]]
            #Iterates through each node in the current layer
            for node in self.NN[i]:
                #Iterates through each value in the inputs and assigns weights
                for j in range(len(inputs)):
                    node.weights[j] += self.learning_rate * node.delta_weight * inputs[j]
                node.weights[-1] += self.learning_rate * node.delta_weight




