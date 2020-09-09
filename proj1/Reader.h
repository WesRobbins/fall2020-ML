//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_READER_H
#define PROJ1_READER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <ctype.h>
#include "DataLine.h"

using namespace std;

// this class is instantiated by DataClass to read data in from file

class Reader {
public:
    string data_name;
    vector<vector<string>> string_data;
    Reader(string file_name);
    vector<vector<string>> file_to_vector(string file_name);
    vector<vector<string>> get_data();
    void print_data();
};


#endif //PROJ1_READER_H
