import math

class Evaluator:
    """A class containing loss functions and percent accuracy function to
    evaluate performance of our model"""

    def __init__(self, classification_type):
        """Initializes the evaluator object with the classification type and performance attributes"""
        self.classification_type = classification_type
        self.performance = 0
        self.num_performances = 0


    def evaluate(self):
        """Runs different loss functions depending on if it is classifying or regressing"""

        if self.classification_type == "classification":
            print(f"Average cross entropy:\t{self.performance / self.num_performances}")
        elif self.classification_type == "regression":
            self.regression_evaluation(test_set, predicted_values)

    def classification_evaluation(self, test_set, predicted_values, certainty):
        """Runs all the classification log functions and prints out their results"""

        percent_accuracy = self.percent_accuracy(test_set, predicted_values)
        one_zero = self.one_zero_loss(test_set, predicted_values)
        log_loss = self.log_loss(test_set, predicted_values, certainty)
        print(f"Percent correct:\t{percent_accuracy * 100:.2f}%")
        print(f"1/0 Loss:\t\t\t{one_zero:.2f}")
        print("Log Loss: ", log_loss)

    def regression_evaluation(self, test_set, predicted_values):
        """Runs all of the regression loss functions and prints out their results"""

        MAE = self.mean_absolute_error(test_set, predicted_values)
        MSE = self.mean_square_error(test_set, predicted_values)
        print(f"Mean Percent Error:\t{MAE:.2f}")
        print(f"Mean Square Error:\t{MSE:.2f}")

    def mean_square_error(self, test_set, predicted_values):
        """Returns the mean square error for a given test set and their predicted values"""

        running_sum = 0
        for i in range(len(test_set)):
            running_sum += (test_set[i].classification - predicted_values[i])**2
        running_sum = running_sum / len(test_set)
        return running_sum

    def cross_entropy(self, truth_labels, predicted_labels):
        #print(f"Truth: {truth_labels}")
        #print(f"Predicted: {predicted_labels}")

        x = sum([(truth_labels[i] * math.log(predicted_labels[i])) for i in range(len(truth_labels))])
        self.performance += x
        self.num_performances += 1
        return x