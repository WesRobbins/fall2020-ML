class Evaluator:
    """A class containing loss functions and percent accuracy function to
    evaluate performance of our model"""

    def __init__(self, classification_type):
        self.classification_type = classification_type

    def evaluate(self, test_set, predicted_values):
        if self.classification_type == "classification":
            self.classification_evaluation(test_set, predicted_values)
        elif self.classification == "regression":
            self.regression_evaluation(test_set, predicted_values)

    def classification_evaluation(self, test_set, predicted_values):
        percent_accuracy = self.percent_accuracy(test_set, predicted_values)
        one_zero = self.one_zero_loss(test_set, predicted_values)
        log_loss = self.log_loss(test_set, predicted_values)
        print(f"Percent correct:\t{percent_accuracy * 100:.2f}%")
        print(f"1/0 Loss:\t\t\t{one_zero:.2f}")

    def regression_evaluation(self, test_set, predicted_values):
        MAE = self.mean_absolute_error(test_set, predicted_values)
        MSE = self.mean_square_error(test_set, predicted_values)
        self.quantile_error(test_set, predicted_values)
        print(f"Mean Percent Error:\t{MAE:.2f}")
        print(f"Mean Square Error:\t{MSE:.2f}")

    def percent_accuracy(self, test_set, predicted_values):
        """Returns percent accuracy given true and predicted values"""

        correct = 0
        for i in range(len(test_set)):
            if (test_set[i].classification == predicted_values[i]):
                correct += 1
        return correct / len(test_set)

    # classification loss functions
    def one_zero_loss(self, test_set, predicted_values):
        """Returns one-zero loss score given true and predicted values"""

        incorrect=0
        for i in range(len(test_set)):
            if(test_set[i].classification != predicted_values[i]):
                incorrect += 1
        return incorrect / len(test_set)

    def log_loss(self, test_set, predicted_values):
        # TODO
        pass

    # regression loss functions
    def mean_absolute_error(self, test_set, predicted_values):
        """Returns the mean absolute error for a given test set and their predicted values"""

        running_sum = 0
        for i in range(len(test_set)):
            running_sum += abs(test_set[i].classification - predicted_values[i])

        running_sum = running_sum / len(test_set)
        return running_sum
    def mean_square_error(self, test_set, predicted_values):
        """Returns the mean square error for a given test set and their predicted values"""

        running_sum = 0
        for i in range(len(test_set)):
            running_sum += (test_set[i].classification - predicted_values[i])**2
        running_sum = running_sum / len(test_set)
        return running_sum

    def quantile_error(self, test_set, predicted_values):
        # TODO
        pass