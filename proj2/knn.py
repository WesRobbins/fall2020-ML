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
        self.hypertune()
        self.train(self.dataclass.df, reduction_type)
        self.edited_data = dataclass.df
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
        # init list of items to delete
        delete = pd.DataFrame()
        # Go through every item, if we predict the incorrect class, delete.
        for index, row in dataframe.iterrows():
            training_set = dataframe.drop(index)
            example = DataLine(row)
            predicted_class = self.classify_example(example, training_set, "classification")
            if predicted_class != example.classification:
                delete.append(row)
        # conduct deletions in a batch after testing each entry on the rest
        # of the data
        out_dataframe = dataframe
        for i, j in delete:
            out_dataframe = dataframe.drop(i)
        return out_dataframe



    def condensed_knn(self, dataframe):
        #initialize the set Z which will eventually become the
        # condensed dataframe output
        Z = []
        # scan all elements of dataframe, adding any
        # values whose nearest neighbor have a different class to Z
        not_done = True
        while not_done:
            z_len = len(Z)
            for index, row in dataframe.iterrows():
                # find distances to random row in dist matrix
                distances_to_r = self.distance_matrix.iloc[index]
                dist_list = list(distances_to_r)
                min_dist = float('inf')
                min_df = pd.DataFrame()
                # find all elements that satisfy the shortest distance to random x
                # who also aren't in the same class as X
                i = 0
                for dist in dist_list:
                    if dist < min_dist:
                        min_dist = dist
                        min_df = dataframe.iloc[i]
                    i += 1

                # if the classes aren't the same, add closest neighbor
                if str(min_df.iloc[-1]) != str(row.iloc[-1]):
                    Z.append(min_df)
            # reduce dataframe to everything that wasn't included earlier
            for index in range(len(Z)):
                dataframe = dataframe.drop(index)
            # No new changes have occurred
            not_done = False
        df_Z = pd.DataFrame(Z)
        # to show reduced dataset, Z
        #print(df_Z)
        return df_Z

    def k_means(self):
        """Using the dataframe, this method selects k random points in our
        vector space to be initial centroids, and then assigns each point
        to its nearest centroid. Each of these assignments forms a set of clusters.
        We then take the mean of each cluster, calling each mean a new centroid.
        Repeat until convergence. (Or for practical purposes, too many iterations
        have passed and we're close enough)"""

        # init k random points
        possible_vals = []
        # hard code k to be used
        means_K = 20
        # find a window of max and min values in the vector space
        max_vect = []
        for i in range(len(self.df.columns)-1):
            column = self.df[i+1]
            # keep track also of all possible values in the vector space, for computing snap-means
            [possible_vals.append(x) for x in column if x not in possible_vals]
            cur_max = column.max()
            max_vect.append(cur_max)

        # also compute the mins in the vector space
        min_vect = []
        for i in range(len(self.df.columns)-1):
            column = self.df[i + 1]
            cur_min = column.min()
            min_vect.append(cur_min)

        # initialize all of the centroids with random values in the vector space on the first
        # iteration
        centroids = []
        for i in range(means_K):
            centroid = []
            for j in range(len(max_vect)+1):
                # give each value a random value out of possible values in original set
                rand_val = rd.randint(0, len(possible_vals)-1)
                centroid.append(possible_vals[rand_val])
            centroids.append(centroid)
        df_centroids = pd.DataFrame(centroids)
        init_df = df_centroids

        cntrd_count = 0

        # after setting up centroids, loop until convergence
        converged = False
        while not converged:
            for index, row in self.df.iterrows():
                example = DataLine(row)
                min_dist = float('inf')
                closest_centroid = pd.DataFrame()
                # find closest centroid for an example
                for index2, row2 in df_centroids.loc[0:, :].iterrows():
                    # Computes the distance between the current point and each centroid
                    cur_dist = self.compute_distance(row.iloc[:-1], row2.iloc[0:])
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_centroid = row2
            # update means of these clusters
            new_mean = []
            for index, row in self.df.iterrows():
                # if example is closest to current centroid, add it to contribute to new mean
                # for this currently considered centroid
                for index2, row2 in df_centroids.loc[0:, :].iterrows():
                    if row2.equals(closest_centroid):
                        off = 0
                        for item in row2:
                            for thing in row:
                                if item != thing:
                                    off += 1
                        if off > (len(row2) / 2):
                            cur_dist = self.compute_distance(row.iloc[:-1], row2.iloc[0:])
                            new_mean.append([row, cur_dist])
            # find new mean for each centroid
            sum = 0
            idx = 0
            for item in new_mean:
                sum += item[1]
                idx += 1
            mean_dist = sum / idx

            # update new centroids
            final_vals = []
            cntrds_out = []
            # for each of the new means we've computed, find the point
            # in the dataframe closest to the mean. (Tried just real valued
            # points in the vector space for this and it didn't work for categorical data,
            # so our model snaps to previous points seen in the space already)
            for item in new_mean:
                item[1] = abs(item[1] - mean_dist)
                final_vals.append(item[1])
            final_vals.sort()
            for i in range(means_K):
                for item in new_mean:
                    if item[1] == final_vals[0]:
                        cntrds_out.append(item[0])
                        final_vals.pop(0)
            df_centroids = pd.DataFrame(cntrds_out)
            for index, row in df_centroids.tail(df_centroids.size - means_K).iterrows():
                df_centroids = df_centroids.drop(index)


            # setup classes for each entry (just so that output is compatible)
            possible_classes = []
            class_column = self.df['Class']
            [possible_classes.append(x) for x in class_column if x not in possible_classes]
            rand_classes = []
            for i in range(means_K):
                if i < len(possible_classes):
                    rand_classes.append(possible_classes[i])
                else:
                    rand_val = rd.randint(0, len(possible_classes)-1)
                    rand_classes.append(possible_classes[rand_val])

            df_centroids['Class'] = rand_classes

            # in case of long lasting runs, 3 iterations is normally close enough to the mean
            if (df_centroids.equals(init_df))or(cntrd_count < 3):
                converged = True
            else:
                cntrd_count += 1
                init_df = df_centroids

            return df_centroids
            
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

        return self.df.loc[medoids]

    def build_cluster_matrix(self, medoids):
        """Builds a distance matrix with the k medoid points and their distances to
        every other point in the dataset"""

        return self.distance_matrix[medoids]

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

