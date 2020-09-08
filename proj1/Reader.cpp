//
// Created by Wes Robbins on 8/26/20.
// Edited by Ben Holmgren

#include "Reader.h"
#include <algorithm>


Reader::Reader(string file_name){
    data_name = file_name.substr(5,5);
    cout<<data_name<<endl;
    string_data = file_to_vector(file_name);

}


vector<vector<string>> Reader::file_to_vector(string file_name){
    vector<vector<string>> data3;
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

        data3.push_back(record);
    }
    if (!infile.eof()){
        cerr << "Fooey!\n";
    }
    if (isalpha(data3[0][0][0])) {
        for (int i = 0; i<data3.size();++i) {
            reverse(data3[i].begin(), data3[i].end());
        }
    }
    return data3;

}

/*vector<DataLine> Reader::vector_to_vector(vector<vector<string>> string_vector){
    vector<DataLine> data2;
    for (vector<string> i : string_vector){

        string classification(i[i.size()-1]);
        i.pop_back();
        vector<float> features;
        if (data_name == "house"){
            for (string j : i){
                if (j == "n"){
                    features.push_back(0);
                }
                else if (j == "y"){
                    features.push_back(1);
                }
                else if (j == "?"){
                    features.push_back(0);
                }
                else {
                    features.push_back(999);
                }
            }
        }
        else if (data_name == "breas"){

        }
        else if (data_name == "glass"){

        }
        else if (data_name == "iris."){

        }
        else if (data_name == "soybe"){

        }
        DataLine dataline(features, classification);
        data2.push_back(dataline);
    }
    return data2;
}
*/
vector<vector<string>> Reader::get_data() {
    return string_data;
}

void Reader::print_data() {
    for (vector<string> i : string_data){
        for (string j : i){
            cout<<j<<" ";
        }
        cout<<endl;
    }
}