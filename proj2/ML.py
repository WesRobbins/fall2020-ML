from DataClass import *
from knn import *

class ML:
    def __init__(self, algorithm_name, c_r, tuning, data_splits, display_settings, file_name):
        self.dataclass = DataClass(file_name, data_splits, c_r)
        self.algorithm = self.initialize_algorithm(algorithm_name, c_r)


        #print(self.dataclass.df.iloc[100,4])




    def initialize_algorithm(self, algorithm_name, classification_type):
        if algorithm_name == "KNN":
            return KNN(self.dataclass, classification_type, "standard")
        elif algorithm_name == "KNNedited":
            return KNN(self.dataclass, classification_type, "edited")
        elif algorithm_name == "KNNcondensed":
            return KNN(self.dataclass, classification_type, "condensed")
        elif algorithm_name == "KNNcluster":
            return KNN(self.dataclass, classification_type, "cluster")