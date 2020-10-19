#from reader import *
#from cross_validation import *
from ML import *
import sys



def main():
    """This is the main function that initializes all machine learning runs. The function instantiates many objects
        of the ML class, each object relating to a different run. The ML class is passed a list of configuration
         settings which are described more in depth below"""

    ML("MLP", "classification", [10, .1], "off",0,0,  "./data/soybean-small.data")

    """ ML params in order:
     1. algorithm name -> options: KNN, KNNedited, KNNcondensed, KNNcluster, KNN_Cluster_Medoids, KNNmenas
     2. classification type -> "classification" or "regression"
     3. data split options ->  [int k, float tuning_set %]
     4. hypertuning on or off -> "on" or "off"
     5. Number of hidden layers
     6. Number of hidden nodes per hidden layer
     7. file name

      """

if __name__ == "__main__":
    main()
