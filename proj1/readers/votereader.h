#ifndef VOTEREADER_H
#define VOTEREADER_H

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int voteread(void) {
        string line;
        ifstream input ("data/house-votes-84.data");
        if (input.is_open()){
                while (getline (input,line)){
                        cout << line << '\n';
                }
                input.close();
        }

        else cout << "somethin's wrong, vern";

        return 0;
}

#endif
