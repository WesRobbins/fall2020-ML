//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_DATACLASS_H
#define PROJ1_DATACLASS_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include "DataLine.h"
#include "Reader.h"

using namespace std;

class DataClass {
public:
    vector<DataLine> data;
    Reader reader;
    DataClass(string file_name);
    vector<vector<DataLine>> single_hold_out(vector<DataLine>);

};


#endif //PROJ1_DATACLASS_H
