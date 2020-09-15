from knn import *
import pandas as pd

class CrossValidation:
    def __init__(self, df):
        """Splits the dataset into k slices, and then runs k experiments using the KNN
        algorithm, switching the training and testing slices each experiment"""

        self.k = 10
        self.data_slices = []
        self.df = df.sample(frac=1)     #This shuffles the data in place
        #self.df = df
        num_rows = self.df.shape[0]
        self.slice_increment = int(num_rows / self.k) #This determines what increment to slice the data in
        self.slice_data()
        self.run_k_experiments()

    def slice_data(self):
        """This method creates a list of k slices of the dataset, to be split
        up into training and testing data"""

        for i in range(self.k):      #This separates the data into k-equal sized slices
            start = i*self.slice_increment
            end = (i+1)*self.slice_increment
            self.data_slices.append(self.df.iloc[start:end, :]) #This is where all the slices are stored


    def run_k_experiments(self):
        """This method runs k experiments on the dataset. It changes which slice is the testing
        set each time and runs it through the KNN algorithm"""

        for i in range(self.k): #This runs the cross validation, using each slice as the testing set
            print(f"Run Number {i+1}:")
            self.testing_set = self.data_slices[i]  #Selects a slice for the testing set

            #Concatenates all slices other than the testing set into the training set
            self.training_set = pd.concat(self.data_slices[:i] + self.data_slices[i+1:])
            KNN(self.training_set, self.testing_set)



