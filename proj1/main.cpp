#include <iostream>
#include <fstream>
#include <string>
//#include "readers/main_reader.h"
//#include "processors/voteprocessor.h"
#include "ML.h"
#include "Reader.h"

using namespace std;

int main(void) {

    //start of new
    ML votes("data/house-votes-84.data");
    //votes.dataclass.reader.print_data();





    // Objects of type a type class include a 2 dimensional vector of data and a vector of

	/*DataClass votes_data (main_reader("data/house-votes-84.data"));
	DataClass breast_cancer_data (main_reader("data/breast_cancer_wisconsin.data"));
	DataClass glass_data (main_reader("data/glass.data"));
	DataClass iris_data (main_reader("data/iris.data"));
	DataClass soybean_data (main_reader("data/soybean-small.data"));
	

	votes_data.print_data(votes_data.get_data());

	cout<<"size of data: "<<votes_data.data.size()<<endl;
	cout<<"size of classification data: "<<votes_data.classifications.size()<<endl;
    */
	/*
	vector <vector <string>> newvotes;
	//newvotes = voteprocess(votes);
	int votesize = votes.size();
	for (int i = 0; i < votesize; i++){
		int datasize = votes[i].size();

		for (int j = 0; j < datasize; j++){
			cout << votes[i][j] << " ";
		}
		cout << "\n" << endl;
	} */
//	cancerread();	
//	glassread();
//	irisread();
//	soyread();
	return 0;
}
