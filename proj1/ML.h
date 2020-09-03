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
    vector<int> bins_count;
    void print_data(vector<DataLine> d);
    void print_title(string file_name);
    void run_all_sets(vector<vector<vector<DataLine>>>);

};


#endif //PROJ1_ML_H
