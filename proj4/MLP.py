from layer import *
from data_line import *
import pandas as pd
from evaluator import *
import sys
import time

class MLP:
    """A class that represents a multi layer perceptron network with a tunable number
    of hidden nodes/ nodes per layer that is capable of binary classification, multi-class
    classification and regression"""

    def __init__(self, dataclass, classification_type, run_network=False):
        """Initializes the Multi Layer Perceptron Class, with a classification type and
        dataframe. Then it sets initial hyperparameters and begins cross-validation experiment"""
        self.c_t = classification_type
        self.dataclass = dataclass
        self.df = dataclass.df  #Initializes the dataset
        self.initialize_parameters()    #Sets initial parameters
        self.eval = Evaluator(self.c_t)
        self.NN = self.initialize_network()

        if run_network:
            self.classify()

    def initialize_parameters(self):
        """Initializes a set of parameters for the neural network"""

        self.n_inputs = len(self.df.columns[:-1])
        self.n_hidden_per_layer = 3
        self.n_hidden = 2
        self.n_outputs = len(self.df.Class.unique()) if self.c_t == "classification" else 1
        self.learning_rate = .07
        self.epochs = 3
        self.momentum_factor = .5
        self.performance = 0

    def hypertune(self):
        """Runs a number of experiments, optimizing k, using the average loss value of the experiments"""

        if self.tuning == "off":
            self.k = 5
        else:
            # if hypertuning is on find optimal k value
            for i in range(5):
                k = i * 2 + 3
                self.k = k
                self.classify()  # Runs the experiment with set k
                avg_performance = self.evaluater.performance / self.evaluater.num_performances
                self.performances.append(avg_performance)
                print("\n Loss score for k = ", self.k, " is ", avg_performance)
            index = self.performances.index(min(self.performances))  # Gets the best performance
            self.k = index * 2 + 3
            self.eval.average_performance()

    def classify(self):
        """Splits the data up into training and testing, then runs k-fold cross validation"""

        data_folds = self.dataclass.make_f_fold("off")
        for i in range(self.dataclass.k):  # This runs the cross validation, using each slice as the testing set
            print(f"Run Number {i + 1}:")
            testing_set = data_folds[i]  # Selects a slice for the testing set

            #  Concatenates all slices other than the testing set into the training set
            training_set = pd.concat(data_folds[:i] + data_folds[i + 1:])
            self.classify_all(training_set, testing_set)
        print("")
        self.eval.average_performance()

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

    def feed_forward(self, inputs):
        """Begins with inputs from the training set, runs it through the neural network and
        returns the output values"""

        #Iterates through each layer, calculating the outputs for each node in
        for layer in self.NN:
            inputs = [node.process_input(inputs) for node in layer]

        #If we are classifying, then compute the softmax values for each node in the output layer
        if self.c_t == "classification":
            inputs = self.output_layer.compute_softmax()

        return inputs

    def backpropagate(self, expected):
        """Backpropagates errors through neural network, assigning a delta weight value to each
        node. This delta weight value is the change that the node will make to its weight"""

        #Assigns delta values to each node in the output layer and calculates momentum
        for i in range(len(self.output_layer)):
            node = self.output_layer[i]
            node.delta_weight = expected[i] - node.output

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

                cur_node.delta_weight = error * cur_node.derivative()

    def update_node_weights(self, inputs):
        """Goes through and updates all the weights utilizing input values, node weights,
        and the learning rate"""

        #Iterates through each node in each layer
        for i in range(len(self.NN)):
            for node in self.NN[i]:
                #Iterates through each value in the inputs and assigns weights
                for j in range(len(inputs)):
                    #Multiplies the weight gradient by the learning rate and input value
                    weight_update = self.learning_rate * node.delta_weight * inputs[j]
                    #Adjusts the weight with momentum
                    node.weights[j] += weight_update + node.momentum[j]
                    #Adjusts the momentum value
                    node.momentum[j] = weight_update
                #Updates the bias node
                node.weights[-1] += self.learning_rate * node.delta_weight
            #Sets the new inputs to the output vector of current layer
            inputs = [node.output for node in self.NN[i]]

    def train(self, training_set):
        """Trains a neural network on a training dataset for a preset number of epochs"""

        for epoch in range(self.epochs):
            #Iterates through a shuffled training set
            for index, row in training_set.sample(frac=1).iterrows():
                input_row = DataLine(row)
                #If classifying, creates a binary encoding of the expected values
                if self.c_t == "classification":
                    expected = [0 for _ in range(self.n_outputs)]
                    expected[int(input_row.classification)] = 1
                #Otherwise, if regressing, creates a list of length 1 of the true value
                else:
                    expected = [input_row.classification for _ in range(self.n_outputs)]

                #Runs the row through the network and adjusts weights
                self.feed_forward(input_row.feature_vector)
                self.backpropagate(expected)
                self.update_node_weights(input_row.feature_vector)


    def test(self, testing_set):
        """Runs a testing set through the neural network and evaluates accuracy"""

        #Creates an evaluator object and creates empty list to hold results
        start_time = time.time()


        true_values = []
        predicted_values = []

        for index, row in testing_set.sample(frac=1).iterrows():
            input_row = DataLine(row)
            if self.c_t == "classification":
                expected = [0] * self.n_outputs
                expected[int(input_row.classification)] = 1
            else:
                expected = [input_row.classification for _ in range(self.n_outputs)]

            outputs = self.feed_forward(input_row.feature_vector)
            true_values.append(expected)
            predicted_values.append(outputs)
        #Evaluates performance of test set
        self.fitness = self.eval.evaluate(true_values, predicted_values)
        end_time = time.time()
        #print(f"Time to initialize class: {end_time - start_time}")
        print(self.fitness)
        return self.fitness

    def classify_all(self, training_set, testing_set):
        """Re initializes the network with random weights, trains it, then tests it"""
        self.NN = self.initialize_network()
        self.train(training_set)
        self.test(testing_set)

    def get_weights(self):
        """Returns on array of all weights in the network for training use"""

        weights = []
        for layer in self.NN:
            for node in layer:
                for weight in node.weights:
                    weights.append(weight)
        return weights

    def set_weights(self, weights):
        """Sets the weights of the nodes in the network after training them"""

        weight_index = 0
        for layer in self.NN:
            for node in layer:
                for i in range(len(node.weights)):
                    #print(weight_index)
                    try:
                        node.weights[i] = weights[weight_index]
                    except Exception as e:
                        print(weight_index)
                        print(len(weights))
                        sys.exit()

                    weight_index += 1






