import sys
import pandas as pd
import operator
import math
from data_line import *
from evaluator import *
from Algorithm import *
import time
from os import path
import copy
import numpy as np

class KNN(Algorithm):
    def __init__(self, dataclass, classification_type, reduction_type, file_name, tuning):
        super(KNN, self).__init__(dataclass, classification_type)
        self.dataclass = dataclass
        self.performances = []
        self.n = 5
        self.k = 5
        self.tuning = tuning
        self.df = self.dataclass.df     # Dataframe that the algorithm will use
        self.vdms = self.dataclass.vdms # List of Value Difference Metrics for each feature
        self.file_name = file_name.split(".")[1]
        # Creates a matrix of the distances between every pairs of values
        self.distance_matrix = self.build_distance_matrix()
        self.edited_data = dataclass.df
        self.hypertune()
        self.train(self.dataclass.df, reduction_type)

        self.classify(self.dataclass)

    def train(self, dataframe, reduction_type):
        if reduction_type == "none":
            pass
        elif reduction_type == "edited":
            self.edited_data = self.edited_knn(dataframe)
        elif reduction_type ==  "condensed":
            self.edited_data = self.condensed_knn(dataframe)
        elif reduction_type == "k_medoids":
            self.edited_testing_set = self.k_medoids()

    def hypertune(self):
        if self.tuning == "off":
            self.k = 5
        else:
            # if hypertuning is on find optimal k value
           for i in range(5):
                k = i*2+3
                self.k = k
                self.classify(self.dataclass)
                avg_performance = self.evaluater.performance / self.evaluater.num_performances
                self.performances.append(avg_performance)
                print("\n Loss score for k = ", self.k, " is ", avg_performance)
           index = self.performances.index(min(self.performances))
           self.k = index*2+3
           print("\nThe optimal k is ", self.k, "\n\n")

    def build_distance_matrix(self):
        """Builds a two dimensional matrix containing the distance between every
        pair of examples"""

        # String that states where the distance matrix can be found/to be created at
        distance_matrix_path = f".{self.file_name}_distance_matrix.data"
        # If the distance matrix has already been created
        if path.isfile(distance_matrix_path):
            # Reads in the distance matrix from the csv
            distance_matrix = pd.read_csv(distance_matrix_path, index_col=0)
            # Converts the column headers from strings to ints
            distance_matrix.columns = distance_matrix.columns.astype(int)
            return distance_matrix
        else:
            indexes = self.df.index # stores the ids of every example in the dataset
            # Creates an n by n matrix, where n is the number of examples in the dataset
            distance_matrix = pd.DataFrame(index=indexes, columns=indexes, dtype="float64")

            for index, row in self.df.iterrows():   # Iterates through every row in the dataset
                # The start_index variable is to avoid calculating the same pair of distances twice
                start_index = index + 1
                # Iterates through every row, after the current one we are looking at, to calculate pair distances
                for index2, row2 in self.df.loc[start_index:,:].iterrows():
                    # Computes the distance between the two examples
                    distance = self.compute_distance(row.iloc[:-1], row2.iloc[:-1])
                    # Populates the distance matrix with the distance
                    distance_matrix.at[index, index2] = distance

            # Transposes and copies the value over the main top left to bottom right diagonal to fill the NaN values
            distance_matrix = distance_matrix.fillna(distance_matrix.transpose(copy=True))

            distance_matrix.to_csv(distance_matrix_path)  # Creates a csv of the distance matrix
            return distance_matrix

    def classify(self, dataclass):
        data_folds = dataclass.make_f_fold(self.edited_data, "off", dataclass.k)
        for i in range(dataclass.k):  #  This runs the cross validation, using each slice as the testing set
            print(f"Run Number {i + 1}:")
            testing_set = data_folds[i]  #  Selects a slice for the testing set

            #  Concatenates all slices other than the testing set into the training set
            training_set = pd.concat(data_folds[:i] + data_folds[i + 1:])
            self.classify_all(training_set, testing_set, dataclass.classification_type)



    def edited_knn(self, dataframe):
        delete = pd.DataFrame()
        for index, row in dataframe.iterrows():
            training_set = dataframe.drop(index)
            example = DataLine(row)
            # print(type(example))
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
        #  conduct deletions in a batch after testing each entry on the rest
        #  of the data
        for i,j in delete:
            self.dataframe = dataframe.drop(i)
            pass
        pass

    def condensed_knn(self, dataframe):
        #  TODO
        pass

    def k_medoids(self):
        """Using the dataframe, this method begins by selecting k random medoids, and
        changing the position of the medoids until distortion is minimized and medoids
        converge"""

        # Selects k random medoids from the dataset and converts their indexes into a list
        medoids = list(self.df.sample(n=self.n).index)
        converged = False   # Initialize boolean as False for the while loop
        # While the medoids have not converged, continue to update them
        while not converged:
            # Copy of medoids to check whether the old medoids and the new ones are different
            old_medoids = copy.deepcopy(medoids)
            #  Builds a matrix of the distance from each medoid to every other point
            cluster_matrix = self.build_cluster_matrix(medoids)
            #  For every point, assigns the medoid it is closest to
            labels = cluster_matrix.idxmin(axis=1)
            # Swaps the medoids out with other datapoints to reduce distortion
            medoids = self.swap_medoids(medoids, labels)
            # If the old medoids and medoids are the same, then the medoids have converged
            converged = old_medoids == medoids

        # Shitty code to test how well the clustering is working
        sum = 0
        for index, row in labels.iteritems():
            predicted_data = DataLine(self.df.loc[row]).classification
            true_data = DataLine(self.df.loc[index])
            if predicted_data == true_data.classification:
                sum += 1
        print(f"PERCENT ACCURACY: {sum / labels.shape[0] * 100}%")

        rows = []   # Empty list to hold the row for each medoid
        for medoid in medoids:
            row = self.df.loc[medoid, :]    # Grab the row in the dataset for that medoid
            rows.append(row)
        # Create a dataframe consisting of only the k medoids
        edited_data = pd.concat(rows, axis=1).transpose()
        return edited_data

    def build_cluster_matrix(self, medoids):
        """Builds a distance matrix with the k medoid points and their distances to
        every other point in the dataset"""

        # Initializes a dataframe of dataset x medoid length
        cluster_matrix = pd.DataFrame(index=self.df.index, columns=medoids, dtype="float64")
        # Iterate through every point in the dataset
        for index in self.df.index:
        # Iterate through each of the k-medoids
            for index2 in medoids:
                # If the indexes are the same, continue since the distance will be 0
                if index == index2:
                    continue
                # Otherwise look up the distance in the distance matrix and populate the cluster matrix
                else:
                    cluster_matrix.at[index, index2] = self.distance_matrix.loc[index, index2]
        return cluster_matrix

    def swap_medoids(self, medoids, labels):
        """Iteratively swaps medoids and random datapoints, keeping a datapoint as the new medoid
        if it lowers the average distortion betweeen datapoints and their medoids"""

        old_medoids = copy.deepcopy(medoids)  # Copy of medoid list to iterate through

        for i in range(len(old_medoids)):
            index = old_medoids[i]          #  Grabs the id of the medoid at i
            cluster_points = labels.loc[labels == index] # Gets all of the datapoints associated with medoid
            best_distortion = self.calculate_distortion(medoids) # Calculates current distortion with medoids
            # Iterates through all of the datapoints within a cluster around a medoid
            for data_point in cluster_points.index:
                if data_point in medoids:     # If the data point is already a medoid, skip this iteration
                    continue
                temp_medoid = medoids[i]      # Store the medoid in case we need to unswap
                medoids[i] = data_point       # Swap the medoids
                new_distortion = self.calculate_distortion(medoids)
                # If the distortion improves, we will keep the medoid and update the best distortion value
                if new_distortion < best_distortion:
                    best_distortion = new_distortion
                else:
                    medoids[i] = temp_medoid    # Unswap medoid back to original
        return medoids

    def calculate_distortion (self, medoids):
        """Calculates the distortion of points around a single medoid"""

        # Builds a cluster distance matrix with the given set of medoids
        medoid_cluster_matrix = self.build_cluster_matrix(medoids)
        # Assigns labels to each datapoint based on which medoid it is closest to
        labels = medoid_cluster_matrix.idxmin(axis=1)
        running_sum = 0

        for data_id, medoid_id in labels.iteritems(): # Iterates through each datapoint and its medoid
            # Looks up the distance between datapoint and medoid, and adds it to the running sum
            running_sum += medoid_cluster_matrix.at[data_id, medoid_id]
        return running_sum

    def compute_distance(self, x, y):
        """Takes in two examples, x and y, and returns
        the multidimensional distance between them"""

        p = 2       # This uses euclidian distance
        d = x.shape[0] # This is the dimensionality of the data
        running_sum = 0     # Keeps a running sum of the distance as we loop through the attributes
        for i in range(d):
            # Checks to see if feature is categorical or real-valued
            if isinstance(x.iloc[i], int) or isinstance(x.iloc[i], float):
                running_sum += (abs(x.iloc[i] - y.iloc[i]))**p  # Minkowski Metric
            else:   # If categorical, looks up the value in corresponding vdm
                col = self.df.columns[i]    # Selects the feature that we are currently looking at
                vdm = self.vdms[col]        # Grabs the vdm for the selected feature
                # Locates the value in the vdm, and adds it to the total distance
                running_sum += vdm.loc[x.iloc[i], y.iloc[i]]

        distance = running_sum**(1/p)
        return distance

    def get_k_neighbors(self, example_id, training_set):
        """Helper function that returns the k nearest neighbors of a given
        example."""

        neighbors = []  # List to hold the nearest neighbors
        # Selects only examples from the training set to be included in the edited matrix
        edited_distance_matrix = self.distance_matrix[self.distance_matrix.index.isin(training_set.index)]

        # Selects the k smallest distances from the edited distance matrix
        neighbor_ids = edited_distance_matrix.nsmallest(self.k, example_id).index
        for id in neighbor_ids: # Iterates through the id of the k nearest neighbors
            # Creates a DataLine of the nearest neighbor and appends it to the neighbors list
            neighbors.append(DataLine(training_set.loc[id,:]))

        return neighbors


    def classify_example(self, example, training_set, classification_type):
        """This computes the distances between a given example and every
        element in the training set and classifies it based on its k-nearest neighbors"""


        example_id = example.feature_vector.name   # The id of the example
        k_nearest_neighbors = self.get_k_neighbors(example_id, training_set) # Returns k nearest neighbors of example

        if classification_type == "classification":
            classes={}

            # This for loop counts the instances of each class within the k-nearest neighbors
            for neighbor in k_nearest_neighbors:

                # Adds counts to a dictionary containing the classes
                if neighbor.classification in classes:
                    classes[neighbor.classification] += 1
                else:
                    classes[neighbor.classification] = 1

            # Returns the class with the most counts

            max1 = max(classes.items(), key=operator.itemgetter(1))[1]
            certainty = max1/self.k
            predicted_class = max(classes.items(), key=operator.itemgetter(1))[0]
            return predicted_class, certainty
        # If performing regression, return a real-valued prediction of the target variable
        elif classification_type == "regression":

            bandwidth = 1       # Gaussian Kernel Bandwidth to be tuned
            dimension = len(example.feature_vector) # Dimensionality of the data
            running_numerator_sum = 0               # Values to hold summation of numerator and denominator
            running_denominator_sum = 0

            for neighbor in k_nearest_neighbors:    # Iterates through the nearest neighbor
                neighbor_id = neighbor.feature_vector.name # Id of the neighbor
                # Looks up the distance between the example and it's neighbor
                distance = self.distance_matrix.loc[example_id, neighbor_id]
                # Calculates a weight by inputting the distance into the kerneling function
                kernel_result = self.kernel_smoother((distance / bandwidth), dimension)
                # Multiplies the weight by the response variable of the neighbor and adds it to numerator sum
                running_numerator_sum += kernel_result * neighbor.classification
                running_denominator_sum += kernel_result

            predicted_class = running_numerator_sum / running_denominator_sum

            return predicted_class, ""

    def kernel_smoother(self, u, dimension):
        """Kernel smoother function"""

        result = (1 / math.sqrt(2 * math.pi)) ** dimension
        result = result * math.exp((-1/2) * (u)**2)
        return result

    def classify_all(self, training_set, testing_set, classification_type):
        """Iterates through the testing set, classifying each example and then calculating
        percent accuracy per testing set"""

        true_values = []
        predicted_values = []
        certainty = []
        for index, row in testing_set.iterrows():
            example = DataLine(row)
            true_values.append(example)
            predicted_class = self.classify_example(example, training_set, classification_type)
            predicted_values.append(predicted_class[0])
            certainty.append(predicted_class[1])

        self.evaluater.evaluate(true_values, predicted_values, certainty)

