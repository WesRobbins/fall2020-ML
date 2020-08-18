#ifndef SOYREADER_H
#define SOYREADER_H

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int soyread(void) {
        string line;
        ifstream input ("data/soybean-small.data");
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

