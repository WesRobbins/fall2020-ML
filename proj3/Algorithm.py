import abc
from evaluator import *


class Algorithm:
    """This class is a Base Class that is inheritted by all of our specific algorithm classes (i.e KNN) and it holds
    the primary attributes and some abstract classes that must be implemented by sub classes"""
    def __init__(self, dataclass, classification_type):
        """Abstract initialization method that sets the dataframe, classification, and an evaluator"""

        self.dataclass = dataclass
        self.c_r = classification_type
        self.evaluater = Evaluator(classification_type)

    @abc.abstractmethod
    def train(self):
        """Abstract training method that trains the method"""
        pass

    @abc.abstractmethod
    def hypertune(self):
        """Abstract method to hypertune parameters"""
        pass

    @abc.abstractmethod
    def classify(self):
        """Abstract method that classifies/regresses examples and splits data into training/testing"""
        pass
