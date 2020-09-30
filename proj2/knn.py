import sys
import pandas as pd
import operator
import math
from data_line import *
from evaluator import *
from Algorithm import *
import time

class KNN(Algorithm):
    def __init__(self, dataclass, classification_type, reduction_type):
        super(KNN, self).__init__(dataclass, classification_type)
        self.df = self.dataclass.df     #Dataframe that the algorithm will use
        self.vdms = self.dataclass.vdms #List of Value Difference Metrics for each feature
        start = time.time()
        self.distance_matrix = self.build_distance_matrix()
        end = time.time()
        print(self.distance_matrix)
        print(f"MATRIX TIME: {end-start}")
        self.edited_data = dataclass.df     #this needs worked with
        self.train(self.dataclass.df, reduction_type)
        self.hypertune()
        start = time.time()
        self.classify(self.dataclass)
        end = time.time()
        print(f"CLASSIFY TIME: {end - start}")

    def train(self, dataframe, reduction_type):
        if reduction_type == "none":
            pass
        elif reduction_type == "edited":
            self.edited_data = self.edited_knn(dataframe)
        elif reduction_type ==  "condensed":
            self.edited_data = self.condensed_knn(dataframe)
        elif reduction_type == "cluster":
            self.edited_data = self.cluster(dataframe)

    def hypertune(self):
        # TODO
        self.k = 3

    def build_distance_matrix(self):
        indexes = self.df.index
        print(indexes)
        distance_matrix = pd.DataFrame(index=indexes, columns=indexes)
        for index, row in self.df.iterrows():
            start_index = index + 1
            for index2, row2 in self.df.loc[start_index:,:].iterrows():
                if index == index2:
                    distance_matrix.at[index, index2] = 0
                    continue
                else:
                    distance = self.compute_distance(row.iloc[:-1], row2.iloc[:-1])
                    distance_matrix.at[index, index2] = distance
        return distance_matrix

    def classify(self, dataclass):
        data_folds = dataclass.make_f_fold(self.edited_data, "on", dataclass.k)
        for i in range(dataclass.k):  # This runs the cross validation, using each slice as the testing set
            print(f"Run Number {i + 1}:")
            testing_set = data_folds[i]  # Selects a slice for the testing set

            # Concatenates all slices other than the testing set into the training set
            training_set = pd.concat(data_folds[:i] + data_folds[i + 1:])
            self.classify_all(training_set, testing_set, dataclass.classification_type)



    def edited_knn(self, dataframe):
        delete = pd.DataFrame()
        for index, row in dataframe.iterrows():
            training_set = dataframe.drop(index)
            example = DataLine(row)
            #print(type(example))
            predicted_class = self.classify_example(example, training_set, "classification")
            if predicted_class != example.classification:
                delete.append(row)
                print(row)
                print("Predicted class")
                print(predicted_class)
                print("GT")
                print(example.classification)
                pass
            pass
        pass
        # conduct deletions in a batch after testing each entry on the rest
        # of the data
        for i,j in delete:
            self.dataframe = dataframe.drop(i)
            pass
        pass

    def condensed_knn(self, dataframe):
        # TODO
        pass

    def cluster(self, dataframe):
        # TODO
        pass



    def compute_distance(self, x, y):
        """Takes in two examples, x and y, and returns
        the multidimensional distance between them"""

        p = 2       #This uses euclidian distance
        d = x.shape[0] #This is the dimensionality of the data
        running_sum = 0     #Keeps a running sum of the distance as we loop through the attributes
        for i in range(d):
            #Checks to see if feature is categorical or real-valued
            if isinstance(x.iloc[i], int) or isinstance(x.iloc[i], float):
                running_sum += (abs(x.iloc[i] - y.iloc[i]))**p  #Minkowski Metric
            else:   #If categorical, looks up the value in corresponding vdm
                col = self.df.columns[i]    #Selects the feature that we are currently looking at
                vdm = self.vdms[col]        #Grabs the vdm for the selected feature
                #Locates the value in the vdm, and adds it to the total distance
                running_sum += vdm.loc[x.iloc[i], y.iloc[i]]

        distance = running_sum**(1/p)
        return distance

    def get_k_neighbors(self, example, training_set):
        """Helper function that returns the k nearest neighbors of a given
        example."""

        distances = []
        neighbors = []  #List to hold the nearest neighbors
        # Computes distance between each example in the training set and the example from the testing set
        for index, row in training_set.iterrows():
            x = DataLine(row)
            distance = self.compute_distance(x.feature_vector, example)
            distances.append((index, distance))  # Stores these distances in the distance list

        distances.sort(key=lambda elem: elem[1])  # Sorts the distances in ascending order

        for distance in distances[:self.k]: #Iterates through the k nearest neighbors and their ids
            id = distance[0]    #Grabs the id that corresponds with the example
            #Creates a DataLine of the nearest neighbor and appends it to the neighbors list
            neighbors.append(DataLine(training_set.loc[id,:]))

        return neighbors


    def classify_example(self, example, training_set, classification_type):
        """This computes the distances between a given example and every
        element in the training set and classifies it based on its k-nearest neighbors"""

        #Grabs the k nearest_neighbors of the example
        #start_time = time.time()
        k_nearest_neighbors = self.get_k_neighbors(example.feature_vector, training_set)
        #end_time = time.time()
        #print(f"get_k_neighbors = {end_time - start_time}")
        if classification_type == "classification":
            classes={}

            #This for loop counts the instances of each class within the k-nearest neighbors
            for neighbor in k_nearest_neighbors:

                #Adds counts to a dictionary containing the classes
                if neighbor.classification in classes:
                    classes[neighbor.classification] += 1
                else:
                    classes[neighbor.classification] = 1

            #Returns the class with the most counts
            predicted_class = max(classes.items(), key=operator.itemgetter(1))[0]

        elif classification_type == "regression":

            bandwidth = 5       #Gaussian Kernel Bandwidth to be tuned
            dimension = len(example.feature_vector)
            running_numerator_sum = 0
            running_denominator_sum = 0
            for neighbor in k_nearest_neighbors:
                distance = self.compute_distance(neighbor.feature_vector, example.feature_vector)
                kernel_result = self.kernel_smoother((distance / bandwidth), dimension)
                running_numerator_sum += kernel_result * example.classification
                running_denominator_sum += self.kernel_smoother((distance / bandwidth), dimension)

            predicted_class = running_numerator_sum / running_denominator_sum

        return predicted_class

    def kernel_smoother(self, u, dimension):

        result = (1 / (2* math.pi)**.5)**dimension
        result = result * math.exp((-1/2) * (u)**2)
        return result

    def classify_all(self, training_set, testing_set, classification_type):
        """Iterates through the testing set, classifying each example and then calculating
        percent accuracy per testing set"""

        true_values = []
        predicted_values = []
        for index, row in testing_set.iterrows():
            example = DataLine(row)
            true_values.append(example)
            predicted_class = self.classify_example(example, training_set, classification_type)
            predicted_values.append(predicted_class)

        self.evaluater.evaluate(true_values, predicted_values)
