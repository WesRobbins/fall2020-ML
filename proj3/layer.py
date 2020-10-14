from node import *

class Layer:
    """Class that represents a layer of nodes in a neural network"""

    def __init__(self, n_nodes, n_inputs):
        """Initializes a list of n_nodes, that have n_inputs going into those nodes"""
        self.nodes = [Node(n_inputs+1) for _ in range(n_nodes)]

    def __repr__(self):
        """Magic method to override string representation in lists"""
        return str(self.nodes)

    def __len__(self):
        """Magic method to represent length of layer"""
        return len(self.nodes)