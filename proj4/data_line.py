class DataLine:
    """This class acts as a data type for the rest of the program. For each data point in stores the
        feature vector as well as its target feature"""
    def __init__(self, row):
        """Assigns all columns up to the end as the vector of attributes and the last column
        as the true class"""

        self.feature_vector = row.iloc[:-1]
        self.classification = row.iloc[-1]
