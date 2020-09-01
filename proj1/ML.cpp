//
// Created by Wes Robbins on 8/26/20.
//

#include "ML.h"

using namespace std;

ML::ML(string file_name)
    :dataclass(file_name) ,
     algorithm()
{
    print_title(file_name);
    vector<vector<DataLine>> train_test = dataclass.single_hold_out(dataclass.data);
    vector<int> bins_count = dataclass.get_bins_count();
    algorithm.run_machine_learning(train_test, bins_count);
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

void ML::print_title(string file_name){
    cout<<"----------------------------------------------\nStarting run on file name: "<<file_name<<"\n----------------------------------------------\n";
}