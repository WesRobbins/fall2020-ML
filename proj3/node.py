from random import *
import math

class Node:
    """A class representing a single neuron in a neural network. Contains info about
    weights coming into this neuron, and is capable of processing inputs and using the
    activation function"""

    def __init__(self, n_inputs):
        """Initializes a random set of weights for each input"""
        self.weights = [random() for _ in range(n_inputs)]

    def process_input(self, inputs):
        """Processes inputs for a node and returns a sum of weights by inputs, utilizing
        the activation function"""

        #Starts by assuming bias node is last weight in weight list
        running_sum = self.weights[-1]
        #Iterates through each weight up to the bias node
        for i in range(len(self.weights) - 1):
            running_sum += self.weights[i] * inputs[i]
        return self.activate(running_sum)

    def activate(self, x):
        """Uses logistic function as activation function"""
        return 1 / (1 + math.exp(x))

    def __repr__(self):
        """Magic method to override string representation in a list"""
        return str(self.weights)

    def __len__(self):
        """Magic method to describe the length of the node"""
        return len(self.weights)