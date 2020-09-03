//
// Created by Wes Robbins on 8/28/20.
//

#ifndef PROJ1_EVALUATE_H
#define PROJ1_EVALUATE_H

#include <math.h>
#include "DataLine.h"
#include "Classifications.h"


class Evaluate {
public:
    vector<DataLine> test_set;
    vector<tuple<Classifications, float>> predicted;
    Evaluate(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in);
    float percent_accuracy(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in);
    float one_zero_loss(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in);
    float log_loss(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in);
    //float precision(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in);
    //float recall(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in);
};


#endif //PROJ1_EVALUATE_H
