#Reads in file path as input and returns preprocessed dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.close('all')

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
class Reader:
    """Creates a dataframe containing the dataset, with changes for each dataset depending on what they need"""


    def __init__(self, file_path):
        """Depending on the dataset chosen, performs dataset specific preprocessing before returning
        the edited dataset to be used by our model"""
        self.df = self.initialize_dataframe(file_path)
        self.remove_constant_features()

    def initialize_dataframe(self, file_path):
        df = pd.read_csv(file_path, header=None)
        if "glass" in file_path:
            df.pop(0)
        elif "forestfires" in file_path:
            df = pd.read_csv(file_path, header=0)
            df.month = df.month.replace(month_dict)
            df["month_sin"] = np.sin(2 * np.pi * df.month / 12)

            df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
            #df["area"] = np.log(df.pop("area"))

        elif "machine" in file_path:
            #Removes the last column in the dataframe
            last_col_index = df.columns[-1]
            df.pop(last_col_index)
        elif "breast-cancer-wisconsin" in file_path:
            df.fillna(df.mean(axis=0), inplace=True)
            df.pop(0)

        return df

    def remove_constant_features(self):
        for col in self.df.columns:
            if len(self.df[col].unique()) == 1:
                self.df.drop(col, inplace=True, axis=1)
        self.df.columns = np.arange(len(self.df.columns))
        self.df.columns = [*self.df.columns[:-1], 'Class']  # Renames the last column class