//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_DATACLASS_H
#define PROJ1_DATACLASS_H
#include <vector>

using namespace std;

class DataClass {
public:
    vector<string> classifications;
    vector<vector<float>> data;

    DataClass(vector<vector<string>>);
    vector<vector<float>> get_data();
    void print_data(vector<vector<float>> data);
};


#endif //PROJ1_DATACLASS_H
