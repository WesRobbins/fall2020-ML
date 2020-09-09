//
// Created by Wes Robbins on 8/28/20.
// Edited by Ben Holmgren

#include "Evaluate.h"
#include <algorithm>

// this class holds functions for evaluating performance
// the methods of this class take in predicted results and the the test data which is referenced for ground truths

// constructor method sets test data and predicted results from model
Evaluate::Evaluate(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in) {
    test_set = test_set_in;
    predicted = predicted_in;
}


// calculates percent accuracy
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

// calculates 1/0 loss
float Evaluate::one_zero_loss(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in) {
    int incorrect_count(0);
    for (int i = 0; i<test_set_in.size(); ++i){
        if (test_set_in[i].classification != get<0>(predicted_in[i]).name){     // if wrong add to incorrect count
            incorrect_count++;
        }
    }
    float loss = (float)incorrect_count/float(test_set_in.size());                  // 1/0 loss score calculation
    return loss;
}

// calculates logg loss
float Evaluate::log_loss(vector<DataLine> test_set_in, vector<tuple<Classifications, float>> predicted_in) {
    float log_totals = 0;
    for (int i = 0; i<test_set_in.size(); ++i){
        if (test_set_in[i].classification != get<0>(predicted_in[i]).name){ // if wrong add ln(1-certainty)
            log_totals += -1*log(1-get<1>(predicted_in[i]));
        }
        else if ((test_set_in[i].classification == get<0>(predicted_in[i]).name)){  // if right add ln(certainty)
            log_totals += -1*log(get<1>(predicted_in[i]));
        }
    }
    float log_score = log_totals/(float)test_set_in.size();                     // averaged to calculate log loss
    return log_score;
}


