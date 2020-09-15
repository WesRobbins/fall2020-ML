class DataLine:
    def __init__(self, row):
        self.feature_vector = row.iloc[:-1]
        self.classification = row.iloc[-1]
