from random import *
import math

class Node:
    """A class representing a single neuron in a neural network. Contains info about
    weights coming into this neuron, and is capable of processing inputs and using the
    activation function"""

    def __init__(self, n_inputs):
        """Initializes a random set of weights for each input"""
        self.weights = [random() for _ in range(n_inputs)]
        self.output = 0
        self.change = 0

    def process_input(self, inputs):
        """Processes inputs for a node and returns a sum of weights by inputs, utilizing
        the activation function"""
        #Starts by assuming bias node is last weight in weight list
        output = self.weights[-1]
        #Iterates through each weight up to the bias node
        for i in range(len(self.weights) - 1):
            output += self.weights[i] * inputs[i]
        self.output = self.activate(output)

    def activate(self, x):
        """Uses hyperbolic tangent function as activation function"""
        return math.tanh(x)

    def derivative(self):
        return 1 - math.tanh(self.output) ** 2

    def __repr__(self):
        """Magic method to override string representation in a list"""
        return str(self.weights)

    def __len__(self):
        """Magic method to describe the length of the node"""
        return len(self.weights)

    def __getitem__(self, key):
        return self.weights[key]