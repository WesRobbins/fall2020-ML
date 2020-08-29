//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_CLASSIFICATIONS_H
#define PROJ1_CLASSIFICATIONS_H
#include <iostream>
#include "DataLine.h"
using namespace std;

class Classifications {
public:

    string name;
    float Q;
    int Nci;
    vector<float> F_vector;
    vector<DataLine> class_data;

    Classifications(string name_in, vector<DataLine> full_train_data);
    vector<DataLine> make_class_data(vector<DataLine> full);
    vector<float> calculate_F_vector(vector<DataLine> c_data);

    vector<float> get_F_vector();

};


#endif //PROJ1_CLASSIFICATIONS_H
