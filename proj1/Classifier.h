//
// Created by Wes Robbins on 8/27/20.
//

#ifndef PROJ1_CLASSIFIER_H
#define PROJ1_CLASSIFIER_H
#include <tuple>
#include "Classifications.h"


class Classifier {
public:
    vector<Classifications> class_vector;
    Classifier();

    tuple<Classifications, float>
    argmax(vector<Classifications> classes, vector<int> input_features, vector<int> vector1);

    float Cx_calculator(Classifications classification, vector<int> input_features, vector<int> bins);

    void set_classes(vector<Classifications> classes_in);


};


#endif //PROJ1_CLASSIFIER_H
