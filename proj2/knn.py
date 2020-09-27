import sys
import pandas as pd
import operator
from data_line import *
from evaluator import *
from Algorithm import *

class KNN(Algorithm):
    def __init__(self, dataclass, classification_type, reduction_type):
        super(KNN, self).__init__(dataclass, classification_type)
        self.df = self.dataclass.df
        self.edited_data = dataclass.df     #this needs worked with
        self.construct_vdm()
        self.train(self.dataclass.df, reduction_type)
        self.hypertune()
        self.classify(self.dataclass)




    def construct_vdm(self):
        """Function to create value difference metrics (vdm) for every column of categorical data."""

        self.vdms = {}          #Create a dictionary that will hold a vdm for each attribute
        df = self.dataclass.df
        classes = df.iloc[:,-1].unique()        #Gets a list of all the classes

        for col in df.iloc[:,:-1]:              #Iterates through each attribute
            #If the current column is not real-valued, construct a vdm for that column
            if df.dtypes[col] != "float64" and df.dtypes[col] != "int64":
                unique_values = df[col].unique()    #Stores all unique feature values
                #Creates a vdm with dimensions of each unique value
                vdm = pd.DataFrame(index=unique_values, columns=unique_values)

                for i in unique_values: #Iterates through the i and j of the vdm
                    for j in unique_values:
                        running_sum = 0
                        for classification in classes:  #Iterates through each class
                            #Computes values for vdm equation
                            cia = self.c_i_a(col, i, classification)
                            ci = self.c_i(col, i)
                            cja = self.c_i_a(col, j, classification)
                            cj = self.c_i(col, j)

                            #Computes value for given class
                            sum = (abs((cia/ci) - (cja/cj)))**1
                            running_sum += sum
                        #Sets the given cell in the vdm to the sum of all classes for a pair (i, j)
                        #of unique, categorical feature vales
                        vdm.at[i,j] = running_sum
                self.vdms[col] = vdm

    def c_i(self, col, feature_value):
        """Helper function that counts the occurrences of a feature value within a column"""

        counts = self.df[col].value_counts()[feature_value]
        return counts

    def c_i_a(self, col, feature_value, classification):
        """Helper function that counts the occurrences of a feature value
        within a column that match a given class"""

        class_df = self.df[self.df['Class'] == classification] #Creates a new df of only examples that match the given class
        counts = class_df[col].value_counts()

        if feature_value not in counts: #If there are no values that correspond with the class
            return 0
        else:
            return counts[feature_value]



            

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

    def classify(self, dataclass):
        data_folds = dataclass.make_f_fold(self.edited_data, "on", dataclass.k)
        for i in range(dataclass.k):  # This runs the cross validation, using each slice as the testing set
            print(f"Run Number {i + 1}:")
            testing_set = data_folds[i]  # Selects a slice for the testing set

            # Concatenates all slices other than the testing set into the training set
            training_set = pd.concat(data_folds[:i] + data_folds[i + 1:])
            self.classify_all(training_set, testing_set, dataclass.classification_type)



    def edited_knn(self, dataframe):
        # TODO
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

    def classify_example(self, example, training_set, classification_type):
        """This computes the distances between a given example and every
        element in the training set and classifies it based on its k-nearest neighbors"""

        distances = []
        #Computes distance between each example in the training set and the example from the testing set
        for index, row in training_set.iterrows():
            x = DataLine(row)
            distance = self.compute_distance(x.feature_vector, example)
            distances.append((index, distance)) #Stores these distances in the distance list

        distances.sort(key=lambda elem: elem[1])    #Sorts the distances in ascending order
        k_nearest_neighbors = distances[:self.k]    #Selects the first k examples from the distances list

        if classification_type == "classification":
            classes={}

            #This for loop counts the instances of each class within the k-nearest neighbors
            for neighbor in k_nearest_neighbors:
                id = neighbor[0]
                neighbor_row= DataLine(training_set.loc[id,:]) #Grabs the row from the training set by id

                #Adds counts to a dictionary containing the classes
                if neighbor_row.classification in classes:
                    classes[neighbor_row.classification] += 1
                else:
                    classes[neighbor_row.classification] = 1

            #Returns the class with the most counts
            predicted_class = max(classes.items(), key=operator.itemgetter(1))[0]

        elif classification_type == "regression":
            # TODO add implentation for regression
            pass


        return predicted_class

    def classify_all(self, training_set, testing_set, classification_type):
        """Iterates through the testing set, classifying each example and then calculating
        percent accuracy per testing set"""

        true_values = []
        predicted_values = []
        total_examples = testing_set.shape[0]
        correct = 0
        for index, row in testing_set.iterrows():
            example = DataLine(row)
            true_values.append(example)
            predicted_class = self.classify_example(example.feature_vector, training_set, classification_type)
            predicted_values.append(predicted_class)

        self.evaluater.evaluate(true_values, predicted_values)








