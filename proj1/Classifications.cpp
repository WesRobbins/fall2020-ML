//
// Created by Wes Robbins on 8/26/20.
// Edited by Ben Holmgren

#include "Classifications.h"
#include <algorithm>

Classifications::Classifications(string name_in, vector<DataLine> full_train_data, vector<int> bins_count) {
    name = name_in;
    class_data = make_class_data(full_train_data);
    Nci = class_data.size();
    Q = (float)Nci/(float)full_train_data.size();

    F_vector = calculate_F_vector(class_data, bins_count);
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

// this function needs adjusted so it can incorporate another dimension on top of
// F_vector to represent the multiple bins. Right now it just represent 1 which can be
// complimented to get 0 for a total of two bins.
// this only works when there is two bins for a feature ( ex: y or n)
vector<vector<float>> Classifications::calculate_F_vector(vector<DataLine> c_data, vector<int> bins_count) {

    // count the amount of each bin in all of the class data
    int max_bins = *max_element(bins_count.begin(), bins_count.end());
    vector<vector<int>> feature_counts{bins_count.size(), vector<int>{max_bins, 0}};

    for (int i = 0; i < feature_counts.size(); i++) {
        for (DataLine j : c_data) {
            feature_counts[i][j.feature_vector[i]]++; //dataline.feature_vector[i] is the bin
        }
    }
    // calculate the % in each bin for each feature
    vector<vector<float>> f_vector;
    for (int i = 0; i < feature_counts.size(); i++) {
        vector<float> feature_Fs;
        for (int j = 0; j<bins_count[i]; j++){
            //**    NAIVE BAYES    **
            float feature_count = feature_counts.size();
            float F{(feature_counts[i][j]  + 1)/ (Nci + feature_count)};
            feature_Fs.push_back(F);
        }

        f_vector.push_back(feature_Fs);
    }
    return f_vector;
}

vector<vector<float>> Classifications::get_F_vector() {
    return F_vector;
}