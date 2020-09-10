//
// Created by Wes Robbins on 8/26/20.
// Edited by Ben Holmgren

#include "Reader.h"
#include <algorithm>

// this class is instantiated by DataClass to read data in from file

Reader::Reader(string file_name){
    data_name = file_name.substr(5,5);
    string_data = file_to_vector(file_name);                        // get data in form of 2d string vector

}

// takes in file name, return string vector
vector<vector<string>> Reader::file_to_vector(string file_name){
    vector<vector<string>> data3;
    ifstream infile(file_name);

    while (infile){                                                 // read file in line by line
        string s;
        if (!getline( infile, s )) break;

        istringstream ss( s );
        vector <string> record;

        while (ss){                                                // read in comma seperated values 1 by 1
            string s;
            if (!getline( ss, s, ',' )) break;
            record.push_back( s );
        }

        data3.push_back(record);
    }
    if (!infile.eof()){
        cerr << "Fooey!\n";
    }

    // this flips the house-votes-84 data around so classification is in the back
    // this is so the data is uniform with all the other data sets which have classification in back
    // for example:     democrat, yes, no, no, yes -> yes, no, no, yes, democrat
    if (isalpha(data3[0][0][0])) {
        for (int i = 0; i<data3.size();++i) {
            reverse(data3[i].begin(), data3[i].end());
        }
    }
    return data3;       // return 2d sreing vector

}

// returns data
vector<vector<string>> Reader::get_data() {
    return string_data;
}


// prints data for examination if necessary
void Reader::print_data() {
    for (vector<string> i : string_data){
        for (string j : i){
            cout<<j<<" ";
        }
        cout<<endl;
    }
}