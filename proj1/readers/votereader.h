#ifndef VOTEREADER_H
#define VOTEREADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>


using namespace std;

vector <vector <string> > voteread(void) {
	
	vector <vector <string> > data;
	ifstream infile( "data/house-votes-84.data" );

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

#endif
