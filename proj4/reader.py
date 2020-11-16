#Reads in file path as input and returns preprocessed dataframe
import pandas as pd
import numpy as np

month_dict = {"jan" : 1,
                  "feb" : 2,
                  "mar" : 3,
                  "apr" : 4,
                  "may" : 5,
                  "jun" : 6,
                  "jul" : 7,
                  "aug" : 8,
                  "sep" : 9,
                  "oct" : 10,
                  "nov" : 11,
                  "dec" : 12}

day_dict = {"sun" : 1,
            "mon" : 2,
            "tue" : 3,
            "wed" : 4,
            "thu" : 5,
            "fri" : 6,
            "sat" : 7,}
class Reader:
    """Creates a dataframe containing the dataset, with changes for each dataset depending on what they need"""


    def __init__(self, file_path):
        """Depending on the dataset chosen, performs dataset specific preprocessing before returning
        the edited dataset to be used by our model"""
        self.df = self.initialize_dataframe(file_path)
        #Removes features that only have 1 unique value.
        self.remove_constant_features()
        self.reset_indices()

    def initialize_dataframe(self, file_path):
        """Performs dataset specific preprocessing on datasets and then converts them into
        pandas dataframes"""
        df = pd.read_csv(file_path, header=None)
        if "glass" in file_path:
            """Deletes columns which have extremely low correlation with class"""
            df.pop(0)
            df.pop(6)
            df.pop(7)

        elif "forestfires" in file_path:
            """Replaces month with two cyclical features to represent
            cyclical nature of months in the year and drops the day column.
            Additionally, performs a log operation on the response variable to counter
            skewing."""
            df = pd.read_csv(file_path, header=0)
            df.month = df.month.replace(month_dict)
            df["month_sin"] = np.sin(2 * np.pi * (df.month / 12))

            df["month_cos"] = np.cos(2 * np.pi * (df.month / 12))
            df.drop("day", axis=1, inplace=True)
            df.drop("month", axis=1, inplace=True)
            df.area = df.area.replace(0, 1e-15)

            df["area"] = np.log(df.pop("area"))

        elif "machine" in file_path:
            """Deletes first two columns since they are irrelevant to performance."""
            #Removes the last column in the dataframe
            last_col_index = df.columns[-1]
            df.pop(last_col_index)
            df.pop(0)
            df.pop(1)

        elif "breast-cancer-wisconsin" in file_path:
            """Fills in each missing value with the mean of their respective column"""
            df[6] = pd.to_numeric(df[6], errors="coerce")
            df.fillna(df.mean(axis=0), inplace=True)
            df.pop(0)
        elif "abalone" in file_path:
            """Replaces sex values with integers corresponding to sex."""
            sex_dictionary = {"M": 1,"F":-1, "I":0}
            df.replace(sex_dictionary, inplace=True)
        return df

    def remove_constant_features(self):
        """Removes all features from the dataset that only have one distinct value"""
        for col in self.df.columns:
            if len(self.df[col].unique()) == 1:
                self.df.drop(col, inplace=True, axis=1)

    def reset_indices(self):
        """Standardizes column headers and indices across each dataframe"""
        self.df.index = np.arange(len(self.df.index))
        self.df.columns = np.arange(len(self.df.columns))
        self.df.columns = [*self.df.columns[:-1], 'Class']  # Renames the last column class