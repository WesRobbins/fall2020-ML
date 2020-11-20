from DataClass import *
from MLP import *
from ga import *

class ML:
    """Takes in the input commands from main and runs the KNN class with the specified algorithm
    and dataset"""
    def __init__(self, algorithm_name, c_r,training_type, data_splits, tuning, file_name):
        """Initializes the ML class with a dataset, file_name and specified algorithm"""

        self.dataclass = DataClass(file_name, data_splits, c_r)
        self.algorithm = self.initialize_algorithm(algorithm_name, c_r, tuning, training_type)



    def initialize_algorithm(self, algorithm_name, classification_type, tuning, training_type):
        """Depending on the algorithm we choose, this method initializes an MLP object with
        a certain algorithm"""
        if training_type == "backpropagation":
            MLP(self.dataclass, classification_type, run_network=True)
        elif training_type == "genetic_algorithm":
            GeneticAlgorithm(self.dataclass, classification_type)
        elif training_type == "differential_evolution":
            DiffEvol(self.dataclass, classification_type)
        elif training_type == "PSO":
            PSO(self.dataclass, classification_type)

        