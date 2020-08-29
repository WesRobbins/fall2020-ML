//
// Created by Wes Robbins on 8/26/20.
//

#include "Classifications.h"

Classifications::Classifications(string name_in, vector<DataLine> full_train_data) {
    name = name_in;
    class_data = make_class_data(full_train_data);
    Nci = class_data.size();
    Q = (float)Nci/(float)full_train_data.size();

    F_vector = calculate_F_vector(class_data);
}


vector<DataLine> Classifications::make_class_data(vector<DataLine> full) {
    vector<DataLine> class_data;
    for (DataLine i : full){
        if (i.classification == name){
            class_data.push_back(i);
        }
    }
    return class_data;
}

// this function needs ajusted so it can incoorporate another dimention on top of
// F_vector to represent the multiple bins. Right now it just represent 1 which can be
// complimented to get 0 for a total of two bins.
// this only works when there is two bins for a feature ( ex: y or n)
vector<float> Classifications::calculate_F_vector(vector<DataLine> c_data) {
    vector<int> feature_counts(c_data[0].feature_vector.size(), 0);

    for (int i = 0; i < feature_counts.size(); i++) {
        for (DataLine j : c_data) {
            if (j.feature_vector[i] == 1) {
                feature_counts[i]++;
            }
        }
    }
    vector<float> f_vector(feature_counts.size(), 0);
    for (int i = 0; i < feature_counts.size(); i++) {
        //**    NAIVE BAYES    **
        float feature_count = feature_counts.size();
        float F{(feature_counts[i]  + 1)/ (Nci + feature_count)};
        f_vector[i] = F;
    }
    return f_vector;
}

vector<float> Classifications::get_F_vector() {
    return F_vector;
}