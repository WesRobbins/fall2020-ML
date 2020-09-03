//
// Created by Wes Robbins on 8/26/20.
//

#include "Algorithm.h"

using namespace std;

Algorithm::Algorithm()
    :classifier()
{


}

void Algorithm::run_machine_learning(vector<vector<DataLine>> train_test_data, vector<int> bins_count_in) {
    bins_count = bins_count_in;
    set_train_test(train_test_data);
    // TRAINING Model
    classes = make_classes(train_set, bins_count);

    // CLASSIFYING test data
    classified_data = get_classified_data(test_set);

    // EVALUATE model
    Evaluate evaluater(test_set, classified_data);
    cout<<"classification accuracy: "<<evaluater.percent_accuracy(evaluater.test_set, evaluater.predicted)*100<<"%"<<endl;
    cout<<"1/0 loss value: "<<evaluater.one_zero_loss(evaluater.test_set, evaluater.predicted)<<endl;
    cout<<"log loss value: "<<evaluater.log_loss(evaluater.test_set, evaluater.predicted)<<endl;
    //print_groundxpredicted(test_set, classified_data);


}

vector<Classifications> Algorithm::make_classes(vector<DataLine> train_in, vector<int> bins_count) {
    vector<string> classes1{};
    for (DataLine i : train_in){
        if (classes1.empty()){
            classes1.push_back(i.classification);
        }
        if (count(classes1.begin(), classes1.end(), i.classification) == 0){

            classes1.push_back(i.classification);

        }
    }
    vector<Classifications> classifications1;
    for (string i : classes1){
        classifications1.push_back(Classifications(i, train_in, bins_count));
    }
    return classifications1;
}

vector<tuple<Classifications, float>> Algorithm::get_classified_data(vector<DataLine> test_data) {
    vector<tuple<Classifications, float>> classified_data1;
    for (int i = 0; test_data.size() > i; i++){
        classified_data1.push_back(classifier.argmax(classes, test_data[i].feature_vector, bins_count));
    }
    return classified_data1;
}

void Algorithm::set_train_test(vector<vector<DataLine>> train_test_data) {
    train_set = train_test_data[0];
    test_set = train_test_data[1];
}


void Algorithm::print_groundxpredicted(vector<DataLine> test_data, vector<tuple<Classifications, float>> predicted) {
    cout<<"\n\nACTUAL CLASSIFICATION VS PREDICTED CLASSIFICATION\n------------------------------\n";
    cout<<"Test Data size is: "<<test_data.size()<<" Predicted Data size is: "<<predicted.size()<<endl;
    for (int i = 0; i<test_data.size(); i++){
        cout<<"Ground: "<<test_data[i].classification<<" Predicted: "<<get<0>(predicted[i]).name<<endl;
        cout<<"Certainty: "<<get<1>(predicted[i])<<endl;
    }
}