import sys
import pandas as pd
import operator
from data_line import *
from evaluator import *

class KNN:
    def __init__(self, training_set, testing_set):
        self.k = 7 #THIS IS THE K-VALUE TO BE EDITED
        self.training_set = training_set
        self.testing_set = testing_set

        self.classify_all()

    def compute_distance(self, x, y):
        """Takes in two examples, x and y, and returns
        the multidimensional distance between them"""

        p = 2       #This uses euclidian distance
        d = x.shape[0] #This is the dimensionality of the data
        running_sum = 0     #Keeps a running sum of the distance as we loop through the attributes
        for i in range(d):
            running_sum += (abs(x.iloc[i] - y.iloc[i]))**p  #Minkowski Metric
        distance = running_sum**(1/p)
        return distance

    def classify_example(self, example):
        """This computes the distances between a given example and every
        element in the training set and classifies it based on its k-nearest neighbors"""

        distances = []
        #Computes distance between each example in the training set and the example from the testing set
        for index, row in self.training_set.iterrows():
            x = DataLine(row)
            distance = self.compute_distance(x.feature_vector, example)
            distances.append((index, distance)) #Stores these distances in the distance list

        distances.sort(key=lambda elem: elem[1])    #Sorts the distances in ascending order
        k_nearest_neighbors = distances[:self.k]    #Selects the first k examples from the distances list
        classes={}

        #This for loop counts the instances of each class within the k-nearest neighbors
        for neighbor in k_nearest_neighbors:
            id = neighbor[0]
            neighbor_row= DataLine(self.training_set.loc[id,:]) #Grabs the row from the training set by id

            #Adds counts to a dictionary containing the classes
            if neighbor_row.classification in classes:
                classes[neighbor_row.classification] += 1
            else:
                classes[neighbor_row.classification] = 1

        #Returns the class with the most counts
        predicted_class = max(classes.items(), key=operator.itemgetter(1))[0]
        return predicted_class

    def classify_all(self):
        """Iterates through the testing set, classifying each example and then calculating
        percent accuracy per testing set"""

        true_values = []
        predicted_values = []
        total_examples = self.testing_set.shape[0]
        correct = 0
        for index, row in self.testing_set.iterrows():
            example = DataLine(row)
            true_values.append(example)
            predicted_class = self.classify_example(example.feature_vector)
            predicted_values.append(predicted_class)

        evaluator = Evaluator(true_values, predicted_values)
        print(f"Percent correct:\t{evaluator.percent_accuracy()*100:.2f}%")
        print(f"1/0 Loss:\t\t\t{evaluator.one_zero_loss():.2f}")







