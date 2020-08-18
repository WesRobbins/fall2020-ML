#ifndef CANCERREADER_H
#define CANCERREADER_H

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int cancerread(void) {
        string line;
        ifstream input ("data/breast-cancer-wisconsin.data");
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
