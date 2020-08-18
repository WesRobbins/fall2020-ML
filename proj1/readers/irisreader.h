#ifndef IRISREADER_H
#define IRISREADER_H

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int irisread(void) {
        string line;
        ifstream input ("data/iris.data");
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

