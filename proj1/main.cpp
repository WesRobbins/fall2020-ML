#include <iostream>
#include <fstream>
#include <string>
#include "readers/votereader.h"
#include "readers/cancerreader.h"

using namespace std;

int main(void) {

	vector <vector <string> > votes;
	votes = voteread();
	int votesize = votes.size();
	for (int i = 0; i < votesize; i++){
		int datasize = votes[i].size();

		for (int j = 0; j < datasize; j++){
			cout << votes[i][j] << " ";
		}
		cout << "\n" << endl;
	}
//	cancerread();	
//	glassread();
//	irisread();
//	soyread();
	return 0;
}
