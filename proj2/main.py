#from reader import *
#from cross_validation import *
from ML import *
import sys



def main():
    """This is the main function that initializes all machine learning runs. The function instantiates many objects
        of the ML class, each object relating to a different run. The ML class is passed a list of configuration
         settings which are described more in depth below"""
    f = open('output.txt', 'w')
    sys.stdout = f


    print("\n\nKNN Classification\n")
    ML("KNN", "classification", "off", [10, .1], 1, "./data/house-votes-84.data")
    print("\n\nKNN Edited Classification\n")
    ML("KNNedited", "classification", "off", [10, .1], 1, "./data/house-votes-84.data")
    print("\n\nKNN Condensed Classification\n")
    ML("KNNcondensed", "classification", "off", [10, .1], 1, "./data/house-votes-84.data")
    print("\n\nKNN Regression\n")
    ML("KNN", "regression", "off", [10, .1], 1, "./data/machine.data")
    print("\n\nKNN Edited Regression")
    ML("KNNedited", "regression", "off", [10, .1], 1, "./data/machine.data")
    print("\n\nKNN Condensed Regression")
    ML("KNNcondensed", "regression", "off", [10, .1], 1, "./data/machine.data")

    f.close()

    """ ML params in order:
     1. algorithm name -> options: KNN, KNNedited, KNNcondensed, KNNcluster, KNN_Cluster_Medoids, KNNmenas
     2. classification type -> "classification" or "regression"
     3. data split options ->  [int k, float tuning_set %]
     4. hypertuning on or off -> "on" or "off"
     4. display option (TODO)-> 0 = k-fold-average only  1 = print all k-folds 2 = also show hypertuning
                                3 = also print data
     5. file name

      """

if __name__ == "__main__":
    main()
