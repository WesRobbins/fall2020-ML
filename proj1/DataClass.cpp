//
// Created by Wes Robbins on 8/26/20.
//

#include "DataClass.h"

using namespace  std;

DataClass::DataClass(string file_name_in)
    :reader(file_name_in)
{
    file_name = file_name_in.substr(5,5);
    data = string_to_DataLine(reader.get_data());
}

vector<DataLine> DataClass::string_to_DataLine(vector<vector<string>> string_data){
    vector<DataLine> final_data;            // this will  be returned from the function
    vector<vector<float>> float_data;
    vector<string> class_names;
    for (vector<string> i : string_data){
        class_names.push_back(i[i.size()-1]);
        //string classification(i[i.size()-1]);
        i.pop_back();
        vector<float> features;
        for (string j : i){
            if (file_name == "house"){

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
            else {
                features.push_back(stof(j));
            }
        }
        float_data.push_back(features);  // adding line to 2d float vector
    }

    // Changing float^2 vector to dataline vector
    tuple<vector<vector<vector<float>>>, vector<int>> bins_in = make_bins(float_data);
    vector<vector<vector<float>>> all_bins_ranges;
    bins_vector = get<1>(bins_in);
    all_bins_ranges = get<0>(bins_in);

    for (int i = 0; i<float_data.size();i++){
        vector<int> classified_features(0);
        for (int j = 0; j<float_data[i].size(); j++){
            for(int k = 0; k<all_bins_ranges[j].size(); k++){
                if (float_data[i][j] >= all_bins_ranges[j][k][0] && float_data[i][j] <= all_bins_ranges[j][k][1]){
                    classified_features.push_back(k);
                    break;
                }
            }

        }

        DataLine dataline(classified_features, class_names[i]);
        final_data.push_back(dataline);
    }
    return final_data;
}

tuple<vector<vector<vector<float>>>, vector<int>> DataClass::make_bins(vector<vector<float>> float_data){
    vector<vector<float>> by_feature_data{float_data[0].size()};
    vector<int> bins;               // this vector will be returned from the function
    for (vector<float> i : float_data){
        for (int j = 0; j<i.size(); j++){
            by_feature_data[j].push_back(i[j]);
        }
    }
    vector<vector<float>> range;
    // range with in each feature
    for (int i = 0; i < by_feature_data.size(); i++){
        vector<float> r(2);
        r[0] = *min_element(by_feature_data[i].begin(), by_feature_data[i].end());
        r[1] = *max_element(by_feature_data[i].begin(), by_feature_data[i].end());
        range.push_back(r);
    }

    // number of bins for each feature
    for (int i = 0; i < by_feature_data.size(); i++){
        bins.push_back(choose_bin_count(range[i]));
    }

    // ranges of bins with in feature
    vector<vector<vector<float>>> all_bins_ranges;
    for (int i = 0; i < by_feature_data.size(); i++){

        vector<vector<float>> per_feature;
        float feature_range = range[i][1]-range[i][0];

        for (int j = 0; j<bins[i]; j++){
            float min = (feature_range/bins[i])*j+range[i][0];
            float max = (feature_range/bins[i]*(j+1))+range[i][0];
            vector<float> bin_range{min, max};
            per_feature.push_back(bin_range);
        }
        all_bins_ranges.push_back(per_feature);
    }


    return {all_bins_ranges, bins};
}

int DataClass::choose_bin_count(vector<float> ranges){
    if (file_name == "house"){
        return 2;
    }
    else {
        return 2;
    }
}

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

vector<vector<DataLine>> DataClass::single_hold_out_nr(vector<DataLine> data_in) {
    vector<vector<DataLine>> train_test(2);
    vector<DataLine> data_copy = data_in;

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

vector<int> DataClass::get_bins_count() {
    return bins_vector;
}
