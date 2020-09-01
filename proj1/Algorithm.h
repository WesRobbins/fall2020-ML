//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_ALGORITHM_H
#define PROJ1_ALGORITHM_H
#include <algorithm>
#include <tuple>
#include "DataLine.h"
#include "Classifications.h"
#include "Classifier.h"
#include "Evaluate.h"

class Algorithm {
public:
    vector<DataLine> train_set;
    vector<DataLine> test_set;
    vector<Classifications> classes;
    vector<int> bins_count;

    vector<tuple<Classifications, float>> classified_data;

    Classifier classifier;

    Algorithm();
    void run_machine_learning(vector<vector<DataLine>> train_test_data, vector<int> bin_count);

    vector<Classifications> make_classes(vector<DataLine> train, vector<int> bins_count);
    vector<tuple<Classifications, float>> get_classified_data(vector<DataLine> test_data);

    void set_train_test(vector<vector<DataLine>> train_test_data);

    void print_groundxpredicted(vector<DataLine> test_data, vector<tuple<Classifications, float>> predicted);
};


#endif //PROJ1_ALGORITHM_H
