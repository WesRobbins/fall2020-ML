import random
import math
import numpy as np

class Node:
    """A class representing a single neuron in a neural network. Contains info about
    weights coming into this neuron, and is capable of processing inputs and using the
    activation function"""

    def __init__(self, n_inputs, type):
        """Initializes a random set of weights for each input"""
        self.weights = [random.uniform(-.01, .01) for _ in range(n_inputs)]
        self.output = 0
        self.type = type
        self.momentum = 0
        self.delta_weight = 0
        self.activate = self.hyperbolic_tangent
        self.derivative = self.hyperbolic_tangent_derivative


    def process_input(self, inputs):
        """Processes inputs for a node and returns a sum of weights by inputs, utilizing
        the activation function"""
        #Starts by assuming bias node is last weight in weight list
        output = self.weights[-1]
        #Iterates through each weight up to the bias node
        for i in range(len(self.weights) - 1):
            output += self.weights[i] * inputs[i]
        if self.type == "output":
            self.output = output
        else:
            self.output = self.activate(output)
    def hyperbolic_tangent(self, x):
        """Uses hyperbolic tangent function as activation function"""
        return math.tanh(x)

    def hyperbolic_tangent_derivative(self,):
        return 1 - (math.tanh(self.output)) ** 2

    def softmax(self, output_vector):
        return math.exp(self.raw_output) / np.sum(np.exp(output_vector))

    def __repr__(self):
        """Magic method to override string representation in a list"""
        repr = f"Weights: {self.weights}\nOutput: {self.output}\nDelta: {self.delta_weight}\n"
        return repr

    def __len__(self):
        """Magic method to describe the length of the node"""
        return len(self.weights)

    def __getitem__(self, key):
        """Magic method to make nodes iterable by their weights"""
        return self.weights[key]