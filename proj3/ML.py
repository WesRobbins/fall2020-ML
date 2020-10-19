from DataClass import *
from MLP import *

class ML:
    """Takes in the input commands from main and runs the KNN class with the specified algorithm
    and dataset"""
    def __init__(self, algorithm_name, c_r, data_splits, tuning, hidden_layers, hidden_nodes, file_name):
        """Initializes the ML class with a dataset, file_name and specified algorithm"""

        self.dataclass = DataClass(file_name, data_splits, c_r)
        self.algorithm = self.initialize_algorithm(algorithm_name, c_r, tuning)



    def initialize_algorithm(self, algorithm_name, classification_type, tuning):
        """Depending on the algorithm we choose, this method initializes an MLP object with
        a certain algorithm"""
        MLP(self.dataclass, classification_type)
        