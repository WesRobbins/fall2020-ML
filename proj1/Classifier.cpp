//
// Created by Wes Robbins on 8/27/20.
//

#include "Classifier.h"

// the Classifier class classifies data from the test set based on the trained model

Classifier::Classifier() {

}

// the argmax function calls Cx calculator to get classification scores and then chooses the class based on a
// classification score
tuple<Classifications, float> Classifier::argmax(vector<Classifications> classes, vector<int> input_feature_vector, vector<int> bins_count) {
    vector<tuple<Classifications, float>> Cx_values;
    for (int i = 0; i<classes.size(); i++){                                     // iterate through all classes
        float Cx = Cx_calculator(classes[i], input_feature_vector, bins_count); // get Cx value
        tuple<Classifications, float> myt{classes[i], Cx};
        Cx_values.push_back(myt);
    }
    int index_of_max;
    vector<float> scores;
    float max = 0.0;
    float total = 0.0;
    for (int i = 0; i < Cx_values.size(); i++){                                 // iterate through cx values to find max
        total += get<1>(Cx_values[i]);
        if (get<1>(Cx_values[i]) > max){
            max = get<1>(Cx_values[i]);
            index_of_max = i;
        }
    }
    float certainty = max/total;                                                // certainty is based on score divided by total possible
    Classifications choosen = get<0>(Cx_values[index_of_max]);
    tuple<Classifications, float> return_values{choosen, certainty};
    return return_values;                                                       // return chosen class and certainty for log loss
}

// this function calculates a classifying value related to the probability that a data line is in a class
float Classifier::Cx_calculator(Classifications classification, vector<int> input_feature_vector, vector<int> bins_count) {
    float Cx = 0;
    float PI_d = 0;
    for (int i = 0; i<bins_count[i]; i++){                          // this for loop sets the accumulative multiplication value to the first
        if (input_feature_vector[0] == i){                          // feature likelihood probability
            PI_d = classification.F_vector[0][i];
            break;
        }
    }
    for (int i = 1; i<input_feature_vector.size(); i++){            // multiply feature likelihood probrability by previous score
        for (int j = 0; j<bins_count[i]; j++){
            if (input_feature_vector[i] == j){
                PI_d *= classification.F_vector[i][j];
                break;
            }
        }
    }
    Cx = PI_d*classification.Q;                                     // get final classification score my multiply by Q of class
    return Cx;
}

// set classes in the classifier vector
void Classifier::set_classes(vector<Classifications> classes_in) {
    class_vector = classes_in;
}