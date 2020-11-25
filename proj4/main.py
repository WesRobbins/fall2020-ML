#from reader import *
#from cross_validation import *
from ML import *
import sys
import time



def main():
    """This is the main function that initializes all machine learning runs. The function instantiates many objects
        of the ML class, each object relating to a different run. The ML class is passed a list of configuration
         settings which are described more in depth below"""


   # ML("MLP", "classification", "genetic_algorithm", [10, .1], "off", "./data/breast-cancer-wisconsin.data")
    ML("MLP", "regression", "genetic_algorithm", [10, .1], "off", "./data/abalone.data")

    # ML("MLP", "classification", "PSO", [10, .1], "off", "./data/breast-cancer-wisconsin.data")

    # ML("MLP", "classification", "differential_evolution", [10, .1], "off", "./data/breast-cancer-wisconsin.data")


    """ ML params in order:
     1. algorithm name -> options: KNN, KNNedited, KNNcondensed, KNNcluster, KNN_Cluster_Medoids, KNNmenas
     2. classification type -> "classification" or "regression"
     3. Training type -> "genetic_algorithm" or "backpropagation"
     4. data split options ->  [int k, float tuning_set %]
     5. hypertuning on or off -> "on" or "off"
     6. Number of hidden layers
     7. Number of hidden nodes per hidden layer
     8. file name

      """

if __name__ == "__main__":
    main()
