//
// Created by Wes Robbins on 8/28/20.
// Edited by Ben Holmgren

#include "Evaluate.h"
#include <algorithm>

Evaluate::Evaluate(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in) {
    test_set = test_set_in;
    predicted = predicted_in;
}

float Evaluate::percent_accuracy(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in) {
    int correct_count(0);
    for (int i = 0; i<test_set_in.size(); ++i){
        if (test_set_in[i].classification == get<0>(predicted_in[i]).name){
            correct_count++;
        }
    }
    float percent = (float)correct_count/float(test_set_in.size());
    return percent;
}

float Evaluate::one_zero_loss(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in) {
    int incorrect_count(0);
    for (int i = 0; i<test_set_in.size(); ++i){
        if (test_set_in[i].classification != get<0>(predicted_in[i]).name){
            incorrect_count++;
        }
    }
    float loss = (float)incorrect_count/float(test_set_in.size());
    return loss;
}

float Evaluate::log_loss(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in) {
    float log_totals = 0;
    for (int i = 0; i<test_set_in.size(); ++i){
        if (test_set_in[i].classification != get<0>(predicted_in[i]).name){
            log_totals += -1*log(1-get<1>(predicted_in[i]));
        }
        else if ((test_set_in[i].classification == get<0>(predicted_in[i]).name)){
            log_totals += -1*log(get<1>(predicted_in[i]));
        }
    }
    float log_score = log_totals/(float)test_set_in.size();
    return log_score;
}


