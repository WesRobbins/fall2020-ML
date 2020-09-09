//
// Created by Wes Robbins on 8/26/20.
// Changed by Ben Holmgren

#ifndef PROJ1_DATACLASS_H
#define PROJ1_DATACLASS_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdlib.h>
#include "DataLine.h"
#include "Reader.h"

using namespace std;

// the DataClass which is instantiated by the ML class is responsible for all reading in,
// descretization, and manipulation of data.

class DataClass {
public:
    string file_name;
    bool noise_on;
    vector<DataLine> data;
    vector<int> bins_vector;
    Reader reader;
    DataClass(string file_name, bool);
    vector<DataLine> string_to_DataLine(vector<vector<string>> string_data);
    tuple<vector<vector<vector<float>>>, vector<int>> make_bins(vector<vector<float>> float_data);
    int choose_bin_count(vector<float> range);
    vector<vector<DataLine>> single_hold_out(vector<DataLine>);
    vector<vector<DataLine>> single_hold_out_nr(vector<DataLine>);
    vector<vector<vector<DataLine>>> ten_fold_cross_validation(vector<DataLine>);

    vector<DataLine> shuffle_feature(vector<DataLine>);

    vector<int> get_bins_count();
};


#endif //PROJ1_DATACLASS_H
