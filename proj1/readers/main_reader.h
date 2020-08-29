//
// Created by Wes Robbins on 8/26/20.
//

#ifndef PROJ1_MAIN_READER_H
#define PROJ1_MAIN_READER_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>


using namespace std;

vector <vector<string>> main_reader(string file_name) {

    vector<vector<string>> data;
    ifstream infile(file_name);

    while (infile){
        string s;
        if (!getline( infile, s )) break;

        istringstream ss( s );
        vector <string> record;

        while (ss){
            string s;
            if (!getline( ss, s, ',' )) break;
            record.push_back( s );
        }

        data.push_back( record );
    }
    if (!infile.eof()){
        cerr << "Fooey!\n";
    }

    return data;

}
#endif //PROJ1_MAIN_READER_H
