#include <iostream>
#include <fstream>
#include <string>
//#include "readers/main_reader.h"
//#include "processors/voteprocessor.h"
#include "ML.h"
#include "Reader.h"

using namespace std;

// the main function in our program instantiates 10 instances of the ML class,
// 2 instances(noise generation on or off) for each of the 5 data sets

int main(void) {

    //ML votes("data/house-votes-84.data", "off");
    //ML glass("data/glass.data", "off");
    //ML iris("data/iris.data", "off");
    ML breast_cancer("data/breast-cancer-wisconsin.data", "off");
    //ML soybean("data/soybean-small.data", "off");
    //------------------------------------------------------
    // Shuffled data models
    //ML votes2("data/house-votes-84.data", "on");
    //ML glass2("data/glass.data", "on");
    //ML iris2("data/iris.data", "on");
    ML breast_cancer2("data/breast-cancer-wisconsin.data", "on");
    //ML soybean2("data/soybean-small.data", "on");

	return 0;
}
