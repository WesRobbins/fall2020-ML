#ifndef GLASSREADER_H
#define GLASSREADER_H

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int glassread(void) {
        string line;
        ifstream input ("data/glass.data");
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

