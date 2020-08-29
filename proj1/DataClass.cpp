//
// Created by Wes Robbins on 8/26/20.
//

#include "DataClass.h"

using namespace  std;

DataClass::DataClass(string file_name)
    :reader(file_name)
{
    data = reader.get_data();
}

vector<vector<DataLine>> DataClass::single_hold_out(vector<DataLine> data_in) {
    vector<vector<DataLine>> train_test(2);
    vector<DataLine> data_copy = data_in;
    auto random_device = std::random_device {};                 //randomize data_copy
    auto rng = std::default_random_engine {random_device()};
    shuffle(std::begin(data_copy), std::end(data_copy), rng);

    for (int i = 0; i<data_copy.size();i++){
        if (i > .8*data_copy.size()){
            train_test[1].push_back(data_copy[i]);
        }
        else{
            train_test[0].push_back(data_copy[i]);
        }
    }
    return train_test;
}