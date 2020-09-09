//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_ML_H
#define PROJ1_ML_H

#include <iostream>
#include "DataClass.h"
#include "Algorithm.h"

using namespace std;

// the ML class is responsible for instantiating the data and then using the data to
// to train and test a model. this is essentially the overarching class for each of the models created

class ML {
public:
    ML(string file_name, string noise);
    Algorithm algorithm;
    vector<int> bins_count;
    void print_data(vector<DataLine> d);
    void print_title(string file_name, bool);
    void run_all_sets(vector<vector<vector<DataLine>>>);
    void print_total_scores(vector<float>);

};


#endif //PROJ1_ML_H
