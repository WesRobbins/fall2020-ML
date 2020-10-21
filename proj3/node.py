import random
import math
import numpy as np

class Node:
    """A class representing a single neuron in a neural network. Contains info about
    weights coming into this neuron, and is capable of processing inputs and using the
    activation function"""

    def __init__(self, n_inputs, node_type):
        """Initializes a random set of weights for each input"""
        self.weights = [random.uniform(-.01, .01) for _ in range(n_inputs)]
        self.output = 0
        self.node_type = node_type
        self.momentum = [0] * n_inputs
        self.delta_weight = 0   #Stores the amount to update node weights

    def process_input(self, inputs):
        """Processes inputs for a node and returns a sum of weights by inputs, utilizing
        the activation function"""

        #Computes the dot product of weights and inputs. then adds the bias node's value
        output = np.dot(self.weights[:-1], inputs) + self.weights[-1]
        #print(inputs)

        #If node is part of hidden layer, sends the output through the activation function
        self.output = output if self.node_type == "output" else self.activate(output)
        return self.output

    def activate(self, x):
        """Uses hyperbolic tangent function as activation function"""
        return math.tanh(x)

    def derivative(self,):
        """Derivative of hyperbolic tangent function, used for gradient descent in
        backpropagation."""
        return 1 - (math.tanh(self.output)) ** 2

    def __repr__(self):
        """Magic method to override string representation in a list"""
        return f"Weights: {self.weights}\nOutput: {self.output}\nDelta: {self.delta_weight}\n"

    def __len__(self):
        """Magic method to describe the length of the node"""
        return len(self.weights)

    def __getitem__(self, key):
        """Magic method to make nodes iterable by their weights"""
        return self.weights[key]