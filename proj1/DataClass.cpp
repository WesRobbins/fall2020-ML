//
// Created by Wes Robbins on 8/26/20.
// Edited by Ben Holmgren

#include "DataClass.h"

using namespace  std;

// the DataClass which is instantiated by the ML class is responsible for all reading in,
// descretization, and manipulation of data.

DataClass::DataClass(string file_name_in, bool noise)
    :reader(file_name_in)                                       //instantiate reader class to read in data
{
    noise_on = noise;
    file_name = file_name_in.substr(5,5);               // format file name into readable form (i.e "data/house-votes--84" -> "house")
    data = string_to_DataLine(reader.get_data());     // use reader class to take from file to vector and string_to_DataLine() to make a data vector

    if (noise_on){                                              // if noise feature is on shuffle 10% of attributes
        data = shuffle_feature(data);
    }
}


// this function turns a vector of strings into a vector of the type DataLine
// * note type DataLine is class that holds a feature vector and classification. more info in DataLine class
vector<DataLine> DataClass::string_to_DataLine(vector<vector<string>> string_data){
    vector<DataLine> final_data;                        // vector of type DataLine will be returned from the function
    vector<vector<float>> float_data;                   // 2d vector of type float will hold attribute vectors and then utilized to make final_data
    vector<string> class_names;                         // vector of type string will hold classifications for each line of data and utilized to make final_data
    for (vector<string> i : string_data){
        class_names.push_back(i[i.size()-1]);           // the last item in the string vectors is classification, this is popped and added to class_names
        i.pop_back();
        vector<float> features;
        for (string j : i){
            if (file_name == "house"){                  // data from house-votes-84 is in form of "y", "n", or "?"
                if (j == "n"){
                    features.push_back(0);
                }
                else if (j == "y"){
                    features.push_back(1);
                }
                else if (j == "?"){
                    features.push_back(0);
                }
                else {
                    features.push_back(999);
                }
            }
            else if (file_name == "breas"){             // data from breast-cancer-wisconsin has "?"
                if (j == "?"){
                    features.push_back(5);
                }
                else {
                    features.push_back(stof(j));        // all other data in breast-cancer-wisconsin can be changed directly from string to float
                }
            }
            else {
                features.push_back(stof(j));            // all other data sets can be changed directly from string to float
            }
        }
        float_data.push_back(features);                 // adding line to 2d float vector
    }


    // the next section of this function descretizes the real valued data
    // and then adds the feature vectors and classification of each data point together to make a DataLine
    tuple<vector<vector<vector<float>>>, vector<int>> bins_in = make_bins(float_data);  //bins_in holds the ranges of all the bins for each attribute
    vector<vector<vector<float>>> all_bins_ranges;
    bins_vector = get<1>(bins_in);                                                   // bins_vector holds the number of bins for each bin
    all_bins_ranges = get<0>(bins_in);                                               // all_bins_ranges holds the ranges of all these bins

    for (int i = 0; i<float_data.size();i++){                                            // if real_value is in bin range, change to corresponding bin (i.e descretize the data)
        vector<int> classified_features(0);
        for (int j = 0; j<float_data[i].size(); j++){
            for(int k = 0; k<all_bins_ranges[j].size(); k++){
                if (float_data[i][j] >= all_bins_ranges[j][k][0] && float_data[i][j] <= all_bins_ranges[j][k][1]){
                    classified_features.push_back(k);
                    break;
                }
            }

        }

        DataLine dataline(classified_features, class_names[i]);                         // form a DataLine that holds classification and feature_vector
        final_data.push_back(dataline);                                                 // push on to final_data
    }
    return final_data;
}


// make_bins function creates bins for real valued data to be put into
// this is done by creating a 2 dimentional vector that is reversely oriented from the original data so that all attributes of
// the same type are grouped together instead of by attributes grouped by each line of data.
tuple<vector<vector<vector<float>>>, vector<int>> DataClass::make_bins(vector<vector<float>> float_data){
    vector<vector<float>> by_feature_data{float_data[0].size()};
    vector<int> bins;                                                                       // this vector will be returned from the function
    for (vector<float> i : float_data){                                                     // reverse orientation (example 5 x 500 to 500 x 5)
        for (int j = 0; j<i.size(); j++){
            by_feature_data[j].push_back(i[j]);
        }
    }
    vector<vector<float>> range;
    // range with in each feature
    for (int i = 0; i < by_feature_data.size(); i++){
        vector<float> r(2);
        r[0] = *min_element(by_feature_data[i].begin(), by_feature_data[i].end());          // find min value for each feature
        r[1] = *max_element(by_feature_data[i].begin(), by_feature_data[i].end());          // find max value for each feature
        range.push_back(r);
    }

    // function is set up so each attribute can have a different number of bins
    for (int i = 0; i < by_feature_data.size(); i++){
        bins.push_back(choose_bin_count(range[i]));
    }

    // ranges of bins with in feature
    // the range of the feature is divided by the number of bins so the ranges of each bin can be determined
    vector<vector<vector<float>>> all_bins_ranges;
    for (int i = 0; i < by_feature_data.size(); i++){

        vector<vector<float>> per_feature;
        float feature_range = range[i][1]-range[i][0];

        for (int j = 0; j<bins[i]; j++){
            float min = (feature_range/bins[i])*j+range[i][0];                              // min bin range
            float max = (feature_range/bins[i]*(j+1))+range[i][0];                          // max bin range
            vector<float> bin_range{min, max};
            per_feature.push_back(bin_range);
        }
        all_bins_ranges.push_back(per_feature);
    }

    // function return all the ranges of bins and also the number of bins for each attribute
    return {all_bins_ranges, bins};
}

// this function is where the number of bins for an attribute are returned
int DataClass::choose_bin_count(vector<float> ranges){
    if (file_name == "house"){
        return 2;
    }
    else {
        return 4;
    }
}

// this function randomizes the data so testing is more accurate
// note: this is not the function that shuffles attributes to make noise
vector<DataLine> randomize(vector<DataLine> original){
    vector<DataLine> data_copy = original;
    auto random_device = std::random_device {};                 //randomize data_copy
    auto rng = std::default_random_engine {random_device()};
    shuffle(std::begin(data_copy), std::end(data_copy), rng);
    return data_copy;
}

// this function returns a training and test set for a single hold out
// experiment to be performed. this was not used to test our hypothesis, but just
// to simplify code development defore 10-fold cross validation was implemented
vector<vector<DataLine>> DataClass::single_hold_out(vector<DataLine> data_in) {
    vector<vector<DataLine>> train_test(2);
    vector<DataLine> data_copy = data_in;
    auto random_device = std::random_device {};                 //randomize data_copy
    auto rng = std::default_random_engine {random_device()};
    shuffle(std::begin(data_copy), std::end(data_copy), rng);

    for (int i = 0; i<data_copy.size();i++){
        if (i > .8*data_copy.size()){
            train_test[1].push_back(data_copy[i]);
        }
        else{
            train_test[0].push_back(data_copy[i]);
        }
    }
    return train_test;
}


// this function splits the data into 10 sub sets in order to form 10 train-test set(90% train, 10% test)
vector<vector<vector<DataLine>>> DataClass::ten_fold_cross_validation(vector<DataLine> all_data){
    all_data = randomize(all_data);
    vector<vector<vector<DataLine>>> return_data;
    vector<vector<DataLine>> ten_fold{10};
    for (int i = 0; i<10; i++){
        for (int j = 0; j<all_data.size(); j++){
            if (j >= .1*i*all_data.size() && j<.1*(i+1)*all_data.size()){
                ten_fold[i].push_back(all_data[j]);
            }
        }
    }
    for (int i = 0; i<10; i++){
        vector<vector<DataLine>> ten_fold_copy{ten_fold};
        vector<vector<DataLine>> train_test{2};
        train_test[1] = ten_fold_copy[i];
        ten_fold_copy.erase(ten_fold_copy.begin()+i);
        vector<DataLine> train;
        for (vector<DataLine> j : ten_fold_copy){
            for (DataLine k : j){
                train.push_back(k);
            }
        }
        train_test[0] = train;
        return_data.push_back(train_test);
    }
    return return_data;
}
// this function shuffles a random feature in the data by extracting the value for a random feature from each data point,
// shuffles those values and re-inserts them into the data
vector<DataLine> DataClass::shuffle_feature(vector<DataLine> data_in){
    vector<DataLine> data_in_copy = data_in;
    int num_features = data_in[0].feature_vector.size();            // number of features
    int random_feature_index = rand() % num_features;               // chooses a random index with in those features
    vector<float> feature_vector;
    for (DataLine i : data_in_copy){                                // push the feature at this index onto 'feature_vector'
        //feature_vector.push_back(i.feature_vector[random_feature_index]);
        feature_vector.push_back(i.feature_vector[0]);
    }
    auto random_device = std::random_device {};                     // shuffle all the values in the feature vector
    auto rng = std::default_random_engine {random_device()};
    shuffle(std::begin(feature_vector), std::end(feature_vector), rng);

    for (int i = 0; i<data_in_copy.size(); i++){                    // insert randomized values into data
        data_in_copy[i].feature_vector[0] = feature_vector[i];
    }
    return data_in_copy;
}


// returns the number of bins of each attribute. used for hyper parameter tuning
vector<int> DataClass::get_bins_count() {
    return bins_vector;
}
