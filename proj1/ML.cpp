//
// Created by Wes Robbins on 8/26/20.
//

#include "ML.h"

using namespace std;

ML::ML(string file_name)
    :dataclass(file_name) ,
     algorithm()
{
    vector<vector<DataLine>> train_test = dataclass.single_hold_out(dataclass.data);
    algorithm.run_machine_learning(train_test);
}

void ML::print_data(vector<DataLine> d) {
    for (DataLine i : d){
        cout<<i.classification<<"    ";
        for (float j : i.feature_vector){
            cout<<j<<" ";
        }
        cout<<endl;
    }
}