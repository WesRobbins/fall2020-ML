//
// Created by Wes Robbins on 8/26/20.
// Edited by Ben Holmgren

#include "Classifications.h"
#include <algorithm>

// this Class is for holding and calculating information regarding each possible classification
// for each instance of this class there is a classification name and a likelihood feature vector


// constructor initiates all functionality of the class
Classifications::Classifications(string name_in, vector<DataLine> full_train_data, vector<int> bins_count) {
    name = name_in;                                             // classification name set
    class_data = make_class_data(full_train_data);              // separate class data out of training data
    Nci = class_data.size();                                    // size of classification (used in Naive Bayes)
    Q = (float)Nci/(float)full_train_data.size();

    F_vector = calculate_F_vector(class_data, bins_count);  // feature likelihood vector
}


// this function separates class data out of all training data which is necessary for NB
vector<DataLine> Classifications::make_class_data(vector<DataLine> full) {
    vector<DataLine> class_data;
    for (DataLine i : full){
        if (i.classification == name){              // if classification of a dataLine equals this.name add to class data
            class_data.push_back(i);
        }
    }
    return class_data;
}

// this function calculates feature likelihood vector
vector<vector<float>> Classifications::calculate_F_vector(vector<DataLine> c_data, vector<int> bins_count) {

    // count the amount of each bin in all of the class data
    int max_bins = *max_element(bins_count.begin(), bins_count.end());
    vector<vector<int>> feature_counts{bins_count.size(), vector<int>{max_bins, 0}};

    for (int i = 0; i < feature_counts.size(); i++) {           // loop through all class data
        for (DataLine j : c_data) {                             // loop through bins of each vector
            feature_counts[i][j.feature_vector[i]]++;           //dataline.feature_vector[i] is the bin
        }
    }
    // calculate the % in each bin for each feature
    vector<vector<float>> f_vector;
    for (int i = 0; i < feature_counts.size(); i++) {
        vector<float> feature_Fs;
        for (int j = 0; j<bins_count[i]; j++){
            //**    NAIVE BAYES    **
            float feature_count = feature_counts.size();
            float F{(feature_counts[i][j]  + 1)/ (Nci + feature_count)};   // NB equation with smoothing
            feature_Fs.push_back(F);
        }

        f_vector.push_back(feature_Fs);
    }
    return f_vector;
}


// returns feature likelihood vector
vector<vector<float>> Classifications::get_F_vector() {
    return F_vector;
}