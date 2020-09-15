#Reads in file path as input and returns preprocessed dataframe
import pandas as pd


class Reader:
    """Creates a dataframe containing the dataset, with changes for each dataset depending on what they need"""

    def __init__(self, file_path):
        if "abalone" in file_path:
            self.df = pd.read_csv(file_path, header=None)

        elif "glass" in file_path:
            self.df = pd.read_csv(file_path, header=None)
            self.df.pop(0)
        elif "house" in file_path:
            #Replaces all question marks with a no vote and
            #moves the classification to the end of the dataframe

            self.df = pd.read_csv(file_path, header=None)
            last_col_index = self.df.columns[-1]
            self.df.replace("?", "n", inplace=True)
            temp_col = self.df.pop(0)
            self.df[last_col_index+1] = temp_col
        elif "machine" in file_path:
            #Removes the last column in the dataframe
            self.df = pd.read_csv(file_path, header=None)
            last_col_index = self.df.columns[-1]
            self.df.pop(last_col_index)
        elif "segmentation" in file_path:
            #Moves the classification column to the end
            self.df = pd.read_csv(file_path)
            temp_col = self.df.pop("CLASS")
            self.df["CLASS"] = temp_col
        else:
            self.df = pd.read_csv(file_path)

