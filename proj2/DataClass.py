from reader import *

"""The data class is responsible for reading in, processing, storing, and manipulating data. An instance of the data
    class is passed into the algorithm so the algorithm has access to all data"""
class DataClass:
    def __init__(self, file_name, data_splits, classification_type):
        self.classification_type = classification_type                      # store classification type (i.e classification or regression)
        self.k = data_splits[0]                                             # number of folds in k fold cross validation
        self.reader = Reader(file_name)                                     # instaniate reader class to read and process file
        self.df = self.reader.df
        self.tuning_set = self.df.iloc[0:int(data_splits[1] * self.df.shape[0]), :]                     #seperate tuning set from data
        self.train_test_set = self.df.iloc[int(data_splits[1] * self.df.shape[0]):self.df.shape[0], :]  # data not in tuning set is train_test data


    def descretize(self, dataframe):
        # TODO
        pass

    def real_valuize(self, dataframe):
        # TODO
        pass


        # stratification is either on or off

    def make_f_fold(self, dataframe, stratification, k):
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
            pass
        return data_folds
