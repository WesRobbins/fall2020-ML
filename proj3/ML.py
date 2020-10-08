from DataClass import *
from knn import *

class ML:
    """Takes in the input commands from main and runs the KNN class with the specified algorithm
    and dataset"""
    def __init__(self, algorithm_name, c_r, tuning, data_splits, display_settings, file_name):
        """Initializes the ML class with a dataset, file_name and specified algorithm"""

        self.dataclass = DataClass(file_name, data_splits, c_r)
        self.file_name = file_name
        self.algorithm = self.initialize_algorithm(algorithm_name, c_r, tuning)



    def initialize_algorithm(self, algorithm_name, classification_type, tuning):
        """Depending on the algorithm we choose, this method initializes a KNN object with
        a certain algorithm"""
        if algorithm_name == "KNN":
            return KNN(self.dataclass, classification_type, "standard", self.file_name, tuning)
        elif algorithm_name == "KNNedited":
            return KNN(self.dataclass, classification_type, "edited", self.file_name, tuning)
        elif algorithm_name == "KNNcondensed":
            return KNN(self.dataclass, classification_type, "condensed", self.file_name, tuning)
        elif algorithm_name == "KNN_Cluster_Medoids":
            return KNN(self.dataclass, classification_type, "k_medoids", self.file_name, tuning)
        elif algorithm_name == "KNNmeans":
            return KNN(self.dataclass, classification_type, "k_means", self.file_name, tuning)
        