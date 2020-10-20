from layer import *
from data_line import *
import numpy as np
import random
import pandas as pd
from evaluator import *
import sys

class MLP:
    """A class that represents a multi layer perceptron network with a tunable number
    of hidden nodes/ nodes per layer that is capable of binary classification, multi-class
    classification and regression"""

    def __init__(self, dataclass, classification_type):
        self.classification_type = classification_type
        self.dataclass = dataclass
        self.df = dataclass.df  #Initializes the dataset
        self.initialize_parameters()    #Sets initial parameters
        self.classify()

    def initialize_parameters(self):
        """Initializes a set of parameters for the neural network"""

        self.n_inputs = len(self.df.columns[:-1])
        self.n_hidden_per_layer = 1
        self.n_hidden = 2
        if self.classification_type == "classification":
            self.n_outputs = len(self.df.Class.unique())
        elif self.classification_type == "regression":
            self.n_outputs = 1
        self.learning_rate = .3
        self.epochs = 3
        self.momentum = .5

    def classify(self):
        """Splits the data up into training and testing, then runs k-fold cross validation"""

        data_folds = self.dataclass.make_f_fold("off")
        for i in range(self.dataclass.k):  # This runs the cross validation, using each slice as the testing set
            print(f"Run Number {i + 1}:")
            testing_set = data_folds[i]  # Selects a slice for the testing set

            #  Concatenates all slices other than the testing set into the training set
            training_set = pd.concat(data_folds[:i] + data_folds[i + 1:])
            self.classify_all(training_set, testing_set)

    def initialize_network(self):
        """Initializes a neural network with an input layer, hidden layers, and output layers"""

        network = []
        #Creates hidden layers
        num_inputs = self.n_inputs
        for layer in range(self.n_hidden):
            hidden_layer = Layer(self.n_hidden_per_layer, num_inputs)
            network.append(hidden_layer)
            num_inputs = len(hidden_layer)
        if self.n_hidden == 0:
            self.output_layer = Layer(self.n_outputs, self.n_inputs, "output")
        else:
            self.output_layer = Layer(self.n_outputs, self.n_hidden_per_layer, "output")
        network.append(self.output_layer)
        return network

    def feed_forward(self, row):
        """Begins with inputs from the training set, runs it through the neural network and
        returns the output values"""

        inputs = row
        for layer in self.NN:
            new_inputs = []
            for node in layer:
                node.process_input(inputs)
                new_inputs.append(node.output)
            inputs = new_inputs


        if self.classification_type == "classification":
            softmax_outputs = self.softmax(inputs)
            new_inputs = []
            for i in range(len(self.output_layer)):
                node = self.output_layer[i]
                node.output = softmax_outputs[i]
                new_inputs.append(node.output)

        inputs = new_inputs
        return inputs

    def backpropagate(self, expected):
        """Backpropagates errors through neural network, assigning a delta weight value to each
        node. This delta weight value is the change that the node will make to its weight"""
        eval = Evaluator(self.classification_type)
        #Assigns delta values to each node in the output layer
        for i in range(len(self.output_layer)):
            node = self.output_layer[i]
            node.momentum = self.momentum * node.delta_weight
            if self.classification_type == "classification":
                node.delta_weight = (expected[i] - node.output)
            else:
                node.delta_weight = expected - node.output

        #Backpropagates errors through hidden layers
        for i in reversed(range(len(self.NN[:-1]))):
            layer = self.NN[i]
            #Iterates through each node in a layer
            for j in range(len(layer)):
                error = 0
                cur_node = layer[j]
                #Iterates through each node in the next layer up
                for node in self.NN[i+1]:

                    error += node.weights[j] * node.delta_weight
                cur_node.momentum = self.momentum * node.delta_weight
                cur_node.delta_weight = error * cur_node.derivative()

    def update_node_weights(self, inputs):
        """Goes through and updates all the weights utilizing input values, node weights,
        and the learning rate"""

        #Iterates through each layer
        for i in range(len(self.NN)):
            #If the layer is not the first layer, use previous layers outputs as inputs
            if i > 0:
                inputs = [node.output for node in self.NN[i-1]]
            #Iterates through each node in the current layer
            for node in self.NN[i]:
                #Iterates through each value in the inputs and assigns weights
                for j in range(len(inputs)):
                    node.weights[j] += self.learning_rate * node.delta_weight * inputs[j] + node.momentum
                node.weights[-1] += self.learning_rate * node.delta_weight + node.momentum

    def train(self, training_set):
        for epoch in range(self.epochs):
            eval = Evaluator(self.classification_type)
            for index, row in training_set.sample(frac=1).iterrows():
                input_row = DataLine(row)
                if self.classification_type == "classification":
                    expected = [0 for _ in range(self.n_outputs)]
                    expected[int(input_row.classification)] = 1
                else:
                    expected = input_row.classification
                outputs = self.feed_forward(input_row.feature_vector)
                eval.MSE(expected, outputs[0])
                self.backpropagate(expected)
                self.update_node_weights(input_row.feature_vector)


    def test(self, testing_set):
        eval = Evaluator(self.classification_type)
        for index, row in testing_set.iterrows():
            input_row = DataLine(row)
            if self.classification_type == "classification":
                expected = [0] * self.n_outputs
                expected[int(input_row.classification)] = 1
            else:
                expected = input_row.classification
            outputs = self.feed_forward(input_row.feature_vector)
            if self.classification_type == "classification":
                eval.cross_entropy(expected, outputs)
                eval.percent_accuracy(expected, outputs)
            else:
                eval.MSE(expected, outputs[0])
        eval.evaluate()

    def classify_all(self, training_set, testing_set):
        self.NN = self.initialize_network()
        self.train(training_set)
        self.test(testing_set)

    def softmax(self, output_vector):
        #print(output_vector)
        exp_vector = np.exp(output_vector)
        return exp_vector / exp_vector.sum()


