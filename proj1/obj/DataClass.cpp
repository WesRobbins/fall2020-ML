//
// Created by Wes Robbins on 8/26/20.
//
#include <iostream>
#include <sstream>
#include "DataClass.h"

using namespace std;

DataClass::DataClass(vector<vector<string>> string_data) {
    for (vector<string> i : string_data) {
        vector<float> data_line;
        for (string j : i) {
            if (j == "republican") {
                classifications.push_back(j);
            } else if (j == "democrat") {
                classifications.push_back(j);
            } else if (j == "y") {
                data_line.push_back(1);
            } else if (j == "n") {
                data_line.push_back(0);
            } else if (j == "?") {
                data_line.push_back(.25);
            } else {
                cout << "Unknown value found in data\n";
            }
        }
        data.push_back(data_line);
    }
}

vector<vector<float>> DataClass::get_data(){
    return data;
}

void DataClass::print_data(vector<vector<float>> data){
    for (int i = 0; i<data.size(); i++){
        cout<<classifications[i]<<" ";
        for (float j : data[i]){
            cout<<j<<" ";
        }
        cout<<endl;
    }
}
