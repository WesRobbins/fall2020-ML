//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_CLASSIFICATIONS_H
#define PROJ1_CLASSIFICATIONS_H
#include <iostream>
#include "DataLine.h"
using namespace std;

// this Class is for holding and calculating information regarding each possible classification
// for each instance of this class there is a classification name and a likelihood feature vector

class Classifications {
public:

    string name;

    float Q;
    int Nci;
    vector<vector<float>> F_vector;
    vector<DataLine> class_data;

    Classifications(string name_in, vector<DataLine> full_train_data, vector<int> bins_data);
    vector<DataLine> make_class_data(vector<DataLine> full);
    vector<vector<float>> calculate_F_vector(vector<DataLine> c_data, vector<int> vector1);

    vector<vector<float>> get_F_vector();

};


#endif //PROJ1_CLASSIFICATIONS_H
