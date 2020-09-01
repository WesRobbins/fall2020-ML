//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_ML_H
#define PROJ1_ML_H

#include <iostream>
#include "DataClass.h"
#include "Algorithm.h"

using namespace std;
class ML {
public:
    ML(string file_name);
    DataClass dataclass;
    Algorithm algorithm;
    void print_data(vector<DataLine> d);
    void print_title(string fil_name);

};


#endif //PROJ1_ML_H
