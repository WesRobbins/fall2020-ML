//
// Created by Wes Robbins on 8/27/20.
//

#include "Classifier.h"


Classifier::Classifier() {

}

tuple<Classifications, float> Classifier::argmax(vector<Classifications> classes, vector<float> input_feature_vector) {
    vector<tuple<Classifications, float>> Cx_values;
    for (int i = 0; i<classes.size(); i++){
        float Cx = Cx_calculator(classes[i], input_feature_vector);
        tuple<Classifications, float> myt{classes[i], Cx};
        Cx_values.push_back(myt);
    }
    int index_of_max;
    vector<float> scores;
    float max = 0.0;
    float total = 0.0;
    for (int i = 0; i < Cx_values.size(); i++){
        total += get<1>(Cx_values[i]);
        if (get<1>(Cx_values[i]) > max){
            max = get<1>(Cx_values[i]);
            index_of_max = i;
        }
    }
    float certainty = max/total;
    Classifications choosen = get<0>(Cx_values[index_of_max]);
    tuple<Classifications, float> return_values{choosen, certainty};
    return return_values;
}

// this is being implemented for 2 bin from 1 vector and complementation
float Classifier::Cx_calculator(Classifications classification, vector<float> input_features) {
    float Cx;
    float PI_d;
    if (input_features[0] == 1){
        PI_d = classification.F_vector[0];
    }
    else if (input_features[0] == 0){
        PI_d = 1-classification.F_vector[0];
    }
    for (int i = 1; i<input_features.size(); i++){
        if (input_features[i] == 1){
            PI_d *= classification.F_vector[i];
        }
        else if (input_features[i] == 0){
            PI_d *= 1-classification.F_vector[i];
        }
    }
    Cx = PI_d*classification.Q;
    return Cx;
}

void Classifier::set_classes(vector<Classifications> classes_in) {
    class_vector = classes_in;
}