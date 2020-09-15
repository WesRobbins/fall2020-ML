from reader import *
from cross_validation import *

def main():
    x = Reader("./data/glass.data") #Reads the data in
    df = x.df       #Creates a dataframe from the reader
    validator = CrossValidation(df)     #Runs the cross validation experiments

if __name__ == "__main__":
    main()