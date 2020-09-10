//
// Created by Wes Robbins on 8/26/20.
//

#include "Algorithm.h"

using namespace std;

// the Algorithm class is given training data and testing data. It is responsible for developing a model form the training
// data and then testing the performance of the algorithm on the test set using loss functions

Algorithm::Algorithm()
    :classifier()                                                   // instantiate Classifier class in constructor
{

}

// this function does the overhead for the training classifying and evaluating
vector<float> Algorithm::run_machine_learning(vector<vector<DataLine>> train_test_data, vector<int> bins_count_in) {
    bins_count = bins_count_in;                                     // bins are passed in so they can be tuned
    set_train_test(train_test_data);
    // TRAINING Model
    // model is trained with training data
    classes = make_classes(train_set, bins_count);  // make classes for each possible classification. each class has name and percent
                                                                    // chance of an attribute being in there classification (preparing to implement NB)
    // CLASSIFYING test data
    // model classifies test_data
    classified_data = get_classified_data(test_set);

    // EVALUATE model
    // performance is evaluated using loss function methods in Evaluater class
    vector<float> scores;                                           // vector to hold scores. index 0 is percent accuracy, index 1 is 1/0 loss, index 2 is log-loss score
    Evaluate evaluater(test_set, classified_data);
    // calculate different perfomance measures and to scores vector
    scores.push_back(evaluater.percent_accuracy(evaluater.test_set, evaluater.predicted)*100);      // percent accuracy
    scores.push_back(evaluater.one_zero_loss(evaluater.test_set, evaluater.predicted));             // 1/0 loss
    scores.push_back(evaluater.log_loss(evaluater.test_set, evaluater.predicted));                  // log loss

    //print_groundxpredicted(evaluater.test_set, evaluater.predicted);
    // show scores in output
    //cout<<"classification accuracy: "<<scores[0]<<"%"<<endl;
    //cout<<"1/0 loss value: "<<scores[1]<<endl;
    //cout<<"log loss value: "<<scores[2]<<endl;

    return scores;

}

// this funciton instantiates a 'Classification' class for each of the possible classification. For example in the votes data
// a Republican and Democrat 'Classification' class is created. Each 'Classification' holds a name and a likelihood attribute
// vector that holds the likelihood that a dataLine with a discrete value is in that class. The calculation of these likilihood
// attribute vectors is a major part of the Naive Bayes implementation
vector<Classifications> Algorithm::make_classes(vector<DataLine> train_in, vector<int> bins_count) {
    vector<string> classes1{};
    for (DataLine i : train_in){                                            // cycle through data and find all classes
        if (classes1.empty()){
            classes1.push_back(i.classification);
        }
        if (count(classes1.begin(), classes1.end(), i.classification) == 0){

            classes1.push_back(i.classification);

        }
    }
    vector<Classifications> classifications1;
    for (string i : classes1){                                              // instantiate a Classification for each possible classification
        classifications1.push_back(Classifications(i, train_in, bins_count));   // add it to a vector
    }
    return classifications1;                                                // return vector of classifications
}


// this function is where the this test data is classified. this is done by utilizing methods in Classifer class
vector<tuple<Classifications, float>> Algorithm::get_classified_data(vector<DataLine> test_data) {
    vector<tuple<Classifications, float>> classified_data1;
    for (int i = 0; test_data.size() > i; i++){                             // utilizing Classifier class classify each line of data
        classified_data1.push_back(classifier.argmax(classes, test_data[i].feature_vector, bins_count));
    }
    return classified_data1;                                                // return a vector of all of the test data classifications
}


// sets training and test data
void Algorithm::set_train_test(vector<vector<DataLine>> train_test_data) {
    train_set = train_test_data[0];
    test_set = train_test_data[1];
}


// a method to print the predicted classifiaction vs ground truth
// only called if more specific evaluation is needed
void Algorithm::print_groundxpredicted(vector<DataLine> test_data, vector<tuple<Classifications, float>> predicted) {
    cout<<"\n\nACTUAL CLASSIFICATION VS PREDICTED CLASSIFICATION\n------------------------------\n";
    cout<<"Test Data size is: "<<test_data.size()<<" Predicted Data size is: "<<predicted.size()<<endl;
    for (int i = 0; i<test_data.size(); i++){
        cout<<"Ground: "<<test_data[i].classification<<" Predicted: "<<get<0>(predicted[i]).name<<endl;
        cout<<"Certainty: "<<get<1>(predicted[i])<<endl;
    }
}