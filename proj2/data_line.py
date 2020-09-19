"""This class acts as a data type for the rest of the program. For each data point in stores the
    feature vector as well as its target feature"""
class DataLine:
    def __init__(self, row):
        self.feature_vector = row.iloc[:-1]
        self.classification = row.iloc[-1]
