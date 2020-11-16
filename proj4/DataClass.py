from reader import *
import pandas as pd


class DataClass:
    """The data class is responsible for reading in, processing, storing, and manipulating data. An instance of the data
        class is passed into the algorithm so the algorithm has access to all data"""
    def __init__(self, file_name, data_splits, classification_type):
        """Initialization method that normalizes all real values and creates value difference
        metrics for the categorical features as well as splitting data into tuning and train/test
        sets."""

        self.k = data_splits[0]                                             #  number of folds in k fold cross validation
        self.reader = Reader(file_name)                                     #  instaniate reader class to read and process file
        self.df = self.reader.df
        self.normalize()    #Normalizes all columns
        if classification_type == "classification":
            self.one_hot_encode()
        self.tuning_set = self.df.iloc[0:int(data_splits[1] * self.df.shape[0]), :] # seperate tuning set from data
        self.train_test_set = self.df.iloc[int(data_splits[1] * self.df.shape[0]):self.df.shape[0], :]  #  data not in tuning set is train_test data

    def normalize(self):
        """Normalizes all real valued features in the dataset"""

        for col in self.df.iloc[:,:-1]:
            if self.df.dtypes[col] == "int64" or self.df.dtypes[col] == "float64":  # Checks to see if real valued
                self.normalize_col(col)

    def normalize_col(self, col):
        """Normalizes a column with z-score normalization, which is raw score - mean score
        divided by the standard deviation"""

        # If there is only 1 value in the feature(constant), set all values to 0.
        if len(self.df[col].unique()) == 1:
            self.df[col] = 0
            return

        col_mean = self.df[col].mean()
        df_std = self.df.std(axis=0)[col]
        self.df[col] = (self.df[col] - col_mean) / df_std

    def make_f_fold(self, stratification):
        """Using the dataframe, this method returns a list of k-length, where each element of the list
        is a dataframe. Basically creates the folds for k-fold cross validation, either through
        stratified sampling, where class frequency is preserved or random sampling."""

        data_folds = []
        num_rows = self.df.shape[0]
        slice_size = int(num_rows / self.k)  # This determines what increment to slice the data in
        self.df.sort_values(by="Class", inplace=True)  # Sorts the examples by their class
        if stratification == "off":
            for i in range(self.k):
                data_folds.append(self.df.iloc[i::self.k,:])
        elif stratification == "on":
            # Creates the folds by including the original frequency of each class within each fold


            # Grabs each class' relative frequency to use for later
            class_proportions = self.df["Class"].value_counts(normalize=True)

            # Iterates through each fold
            for i in range(self.k):
                single_fold = []    # Creates an empty list to be populated

                # Iterates through each class in the dataframe
                for classification, frame in self.df.groupby(by="Class"):
                    single_class_proportion = class_proportions[classification] # Grabs the class frequency

                    # Determines how many examples to put in this fold, based on frequency and fold size
                    examples_for_fold = single_class_proportion * slice_size

                    # Creates the starting and ending indices
                    start = int(i * examples_for_fold)
                    end = int((i + 1) * examples_for_fold)

                    # Appends x number of examples from this class into this fold
                    single_fold.append(frame.iloc[start:end, :])
                # Concatenates the list of dataframes together into one dataframe
                new_dataframe = pd.concat(single_fold)

                # Appends the dataframe(one fold) to the list of folds
                data_folds.append(new_dataframe)
        return data_folds

    def one_hot_encode(self):
        """Transforms the class label into the index of a 1 in a one hot encoding scheme"""
        classes = list(self.df.Class.unique())

        for index, row in self.df.iterrows():
            classification = row.loc["Class"]
            one_hot_index = classes.index(classification)
            self.df.loc[index, "Class"] = one_hot_index
