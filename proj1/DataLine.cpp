//
// Created by Wes Robbins on 8/26/20.
//

#include "DataLine.h"


DataLine::DataLine(vector<float> feature_vector_in, string classification_in) {
    feature_vector = feature_vector_in;
    classification = classification_in;
}

string DataLine::get_classification() {
    return classification;
}

vector<float> DataLine::get_feature_vector() {
    return feature_vector;
}