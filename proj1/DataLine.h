//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_DATALINE_H
#define PROJ1_DATALINE_H

#include <vector>
#include <iostream>

using namespace std;

class DataLine {
public:
    vector<int> feature_vector;
    string classification;
    DataLine(vector<int> feature_vector_in, string classification_in);
    string get_classification();
    vector<int> get_feature_vector();
};


#endif //PROJ1_DATALINE_H
