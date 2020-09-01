//
// Created by Wes Robbins on 8/27/20.
//

#include "Classifier.h"


Classifier::Classifier() {

}

tuple<Classifications, float>
Classifier::argmax(vector<Classifications> classes, vector<int> input_feature_vector, vector<int> bins_count) {
    vector<tuple<Classifications, float>> Cx_values;
    for (int i = 0; i<classes.size(); i++){
        float Cx = Cx_calculator(classes[i], input_feature_vector, bins_count);
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
float Classifier::Cx_calculator(Classifications classification, vector<int> input_feature_vector, vector<int> bins_count) {
    float Cx = 0;
    float PI_d = 0;
    for (int i = 0; i<bins_count[i]; i++){
        if (input_feature_vector[0] == i){
            PI_d = classification.F_vector[0][i];
            break;
        }
    }
    for (int i = 1; i<input_feature_vector.size(); i++){
        for (int j = 0; j<bins_count[i]; j++){
            if (input_feature_vector[i] == j){
                PI_d *= classification.F_vector[i][j];
                break;
            }
        }
    }
    Cx = PI_d*classification.Q;
    return Cx;
}

void Classifier::set_classes(vector<Classifications> classes_in) {
    class_vector = classes_in;
}