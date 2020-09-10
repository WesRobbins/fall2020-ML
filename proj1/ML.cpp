//
// Created by Wes Robbins on 8/26/20.
//

#include "ML.h"

using namespace std;

// the ML class is responsible for instantiating the data and then using the data to
// to train and test a model. this is essentially the overarching class for each of the models created

ML::ML(string file_name, string noise)
    :algorithm()
{
    bool noise_on;                                          // configure if noise generation is on or off
    if (noise == "on"){
        noise_on = true;
    }
    else if (noise == "off") {
        noise_on = false;
    }
    DataClass dataclass(file_name, noise_on);               // instantiate data class
    print_title(file_name, noise_on);
    vector<vector<vector<DataLine>>> ten_fold = dataclass.ten_fold_cross_validation(dataclass.data);    //get data for 10-fold cross validation
    bins_count = dataclass.get_bins_count();                // get number of bins so hyper parameter testing can be performed
    run_all_sets(ten_fold);                                 // run each of the 10 train-test sets in cross validation and also measure performance
    //algorithm.run_machine_learning(ten_fold[0], bins_count);
}


// this function calls Algorithm class method for each of the train-test sets
void ML::run_all_sets(vector<vector<vector<DataLine>>> ten_fold){
    vector<float> scores_total{0,0,0};
    for (vector<vector<DataLine>> i : ten_fold){
        vector<float> scores = algorithm.run_machine_learning(i, bins_count);
        for (int j = 0; j<scores.size(); j++){
            scores_total[j] += scores[j];
        }
    }
    for (int i = 0; i<scores_total.size(); i++){
        scores_total[i] = scores_total[i]/ten_fold.size();
    }
    print_total_scores(scores_total);
}


// prints data for testing and debugging
void ML::print_data(vector<DataLine> d) {
    for (DataLine i : d){
        cout<<i.classification<<"    ";
        for (float j : i.feature_vector){
            cout<<j<<" ";
        }
        cout<<endl;
    }
}


// prints title of data name to increase readability of output
void ML::print_title(string file_name, bool noise_on){
    cout<<"----------------------------------------------\nStarting run on file name: "<<file_name<<endl;
    if (noise_on){
        cout<<"Noise generation is on";
    }
    else {
        cout<<"Noise generation is off";
    }
    cout<<"\n----------------------------------------------\n";
}

void ML::print_total_scores(vector<float> scores){
    cout<<"\nAVERAGE SCORES FROM 10 FOLD CROSS VALIDATION:\n";
    cout<<"percent accuracy: "<<scores[0]<<endl;
    cout<<"1/0 loss: "<<scores[1]<<endl;
    cout<<"log loss: "<<scores[2]<<endl;
}