from DataClass import *
from knn import *

class ML:
    def __init__(self, algorithm_name, c_r, tuning, data_splits, display_settings, file_name):
        self.dataclass = DataClass(file_name, data_splits, c_r)
        self.file_name = file_name
        self.algorithm = self.initialize_algorithm(algorithm_name, c_r)



    def initialize_algorithm(self, algorithm_name, classification_type):
        if algorithm_name == "KNN":
            return KNN(self.dataclass, classification_type, "standard", self.file_name)
        elif algorithm_name == "KNNedited":
            return KNN(self.dataclass, classification_type, "edited", self.file_name)
        elif algorithm_name == "KNNcondensed":
            return KNN(self.dataclass, classification_type, "condensed", self.file_name)
        elif algorithm_name == "KNNcluster":
            return KNN(self.dataclass, classification_type, "cluster", self.file_name)