from reader import *
import pandas as pd

"""The data class is responsible for reading in, processing, storing, and manipulating data. An instance of the data
    class is passed into the algorithm so the algorithm has access to all data"""
class DataClass:
    def __init__(self, file_name, data_splits, classification_type):
        self.classification_type = classification_type                      # store classification type (i.e classification or regression)
        self.k = data_splits[0]                                             # number of folds in k fold cross validation
        self.reader = Reader(file_name)                                     # instaniate reader class to read and process file
        self.df = self.reader.df
        self.normalize()
        self.vdms = self.construct_vdm()
        self.tuning_set = self.df.iloc[0:int(data_splits[1] * self.df.shape[0]), :]                     #seperate tuning set from data
        self.train_test_set = self.df.iloc[int(data_splits[1] * self.df.shape[0]):self.df.shape[0], :]  # data not in tuning set is train_test data


    def normalize(self):
        """Normalizes all real valued features in the dataset"""

        for col in self.df.iloc[:,:-1]:
            if self.df.dtypes[col] == "int64" or self.df.dtypes[col] == "float64":  #Checks to see if real valued
                self.normalize_col(col)

    def normalize_col(self, col):
        """Normalizes a column with z-score normalization, which is raw score - mean score
        divided by the standard deviation"""

        #If there is only 1 value in the feature(constant), set all values to 0.
        if len(self.df[col].unique()) == 1:
            self.df[col] = 0
            return

        col_mean = self.df[col].mean()
        df_std = self.df.std(axis=0)[col]
        self.df[col] = (self.df[col] - col_mean) / df_std

    def descretize(self, dataframe):
        # TODO
        pass

    def construct_vdm(self):
        """Function to create value difference metrics (vdm) for every column of categorical data."""

        vdms = {}  # Create a dictionary that will hold a vdm for each attribute
        classes = self.df.iloc[:, -1].unique()  # Gets a list of all the classes

        for col in self.df.iloc[:, :-1]:  # Iterates through each attribute
            # If the current column is not real-valued, construct a vdm for that column
            if self.df.dtypes[col] != "float64" and self.df.dtypes[col] != "int64":
                unique_values = self.df[col].unique()  # Stores all unique feature values
                # Creates a vdm with dimensions of each unique value
                vdm = pd.DataFrame(index=unique_values, columns=unique_values)

                for i in unique_values:  # Iterates through the i and j of the vdm
                    for j in unique_values:
                        if i == j:
                            vdm.at[i, j] = 0
                            continue
                        running_sum = 0
                        #If classification type is regression, just compute raw difference in frequencies
                        if self.classification_type == "regression":
                            #Calculates the counts divided by total rows to get frequency
                            ci = self.c_i(col, i) / self.df.shape[0]
                            cj = self.c_i(col, j) / self.df.shape[0]
                            vdm.at[i, j] = (abs(ci - cj))
                            continue
                        for classification in classes:  # Iterates through each class
                            # Computes values for vdm equation
                            cia = self.c_i_a(col, i, classification)
                            ci = self.c_i(col, i)
                            cja = self.c_i_a(col, j, classification)
                            cj = self.c_i(col, j)

                            # Computes value for given class
                            sum = (abs((cia / ci) - (cja / cj)))
                            running_sum += sum
                        # Sets the given cell in the vdm to the sum of all classes for a pair (i, j)
                        # of unique, categorical feature vales
                        vdm.at[i, j] = running_sum
                vdms[col] = vdm

        return vdms

    def c_i(self, col, feature_value):
        """Helper function that counts the occurrences of a feature value within a column"""

        counts = self.df[col].value_counts()[feature_value]
        return counts

    def c_i_a(self, col, feature_value, classification):
        """Helper function that counts the occurrences of a feature value
        within a column that match a given class"""

        class_df = self.df[
            self.df['Class'] == classification]  # Creates a new df of only examples that match the given class
        counts = class_df[col].value_counts()

        if feature_value not in counts:  # If there are no values that correspond with the class
            return 0
        else:
            return counts[feature_value]

        # stratification is either on or off

    def make_f_fold(self, dataframe, stratification, k):
        """Using the dataframe, this method returns a list of k-length, where each element of the list
        is a dataframe. Basically creates the folds for k-fold cross validation, either through
        stratified sampling, where class frequency is preserved or random sampling."""

        data_folds = []
        num_rows = dataframe.shape[0]
        slice_size = int(num_rows / k)  # This determines what increment to slice the data in
        if stratification == "off":
            dataframe = dataframe.sample(frac=1)  # This shuffles the data in place
            for i in range(k):
                start = i * slice_size
                end = (i + 1) * slice_size
                data_folds.append(dataframe.iloc[start:end, :])
        elif stratification == "on":
            #Creates the folds by including the original frequency of each class within each fold
            dataframe.sort_values(by="Class", inplace=True) #Sorts the examples by their class

            #Grabs each class' relative frequency to use for later
            class_proportions = dataframe["Class"].value_counts(normalize=True)

            #Iterates through each fold
            for i in range(self.k):
                single_fold = []    #Creates an empty list to be populated

                #Iterates through each class in the dataframe
                for classification, frame in self.df.groupby(by="Class"):
                    single_class_proportion = class_proportions[classification] #Grabs the class frequency

                    #Determines how many examples to put in this fold, based on frequency and fold size
                    examples_for_fold = single_class_proportion * slice_size

                    #Creates the starting and ending indices
                    start = int(i * examples_for_fold)
                    end = int((i + 1) * examples_for_fold)

                    #Appends x number of examples from this class into this fold
                    single_fold.append(frame.iloc[start:end, :])
                #Concatenates the list of dataframes together into one dataframe
                new_dataframe = pd.concat(single_fold)

                #Appends the dataframe(one fold) to the list of folds
                data_folds.append(new_dataframe)
        return data_folds
