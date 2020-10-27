import math
import numpy as np

class Evaluator:
    """A class containing loss functions and percent accuracy function to
    evaluate performance of our model"""

    def __init__(self, classification_type):
        """Initializes the evaluator object with the classification type and performance attributes"""
        self.classification_type = classification_type

    def evaluate(self, true_values, predicted_values):
        """Runs different loss functions depending on if it is classifying or regressing"""

        if self.classification_type == "classification":
            self.cross_entropy(true_values, predicted_values)
            self.percent_accuracy(true_values,predicted_values)
        elif self.classification_type == "regression":
            self.mean_squared_error(true_values, predicted_values)
            self.mean_absolute_error(true_values, predicted_values)

    def cross_entropy(self, true_values, predicted_values):
        """Calculates the average cross entropy across a testing set and prints out that
        information"""

        testing_set_size = len(true_values)
        running_sum = 0
        for i in range(len(true_values)):
            true_set = true_values[i]
            predicted_set = predicted_values[i]
            running_sum = sum([(true_set[j] * math.log(predicted_set[j])) for j in range(len(true_set))])

        print(f"Average cross entropy:\t{-running_sum / testing_set_size}")
        return -running_sum / testing_set_size

    def percent_accuracy(self, true_values, predicted_values):
        """Calculates the average percent accuracy across classification of a testing set
        and prints out that information"""

        correct = 0
        size = len(true_values)
        for i in range(len(true_values)):
            true_labels = true_values[i]
            predicted_labels = predicted_values[i]
            predicted_index = np.argmax(predicted_labels)

            if true_labels[predicted_index] == 1:
                correct += 1
        print(f"Percent Accuracy:\t\t{correct / size * 100:.2f}%")

    def mean_squared_error(self, true_values, predicted_values):
        """Calculates the mean squared error of predictions over a testing set"""

        size = len(true_values)
        running_sum = 0
        for i in range(len(true_values)):
            running_sum += (true_values[i][0] - predicted_values[i][0]) ** 2
        print(f"Mean Squared Error:\t{running_sum/size}")

    def mean_absolute_error(self, true_values, predicted_values):
        """Calculates the mean absolute error of predictions over a testing set"""

        size = len(true_values)
        running_sum = 0
        for i in range(len(true_values)):
            running_sum += abs(true_values[i][0] - predicted_values[i][0])
        print(f"Mean Absolute Error:\t{running_sum/size}")

