import abc
from evaluator import *

"""This class is a Base Class that is inheritted by all of our specific algorithm classes (i.e KNN) and it holds
    the primary attributes and some abstract classes that must be implemented by sub classes"""
class Algorithm:

    def __init__(self, dataclass, classification_type):
        self.dataclass = dataclass
        self.c_r = classification_type
        self.evaluater = Evaluator(classification_type)

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def hypertune(self):
        pass

    @abc.abstractmethod
    def classify(self):
        pass
