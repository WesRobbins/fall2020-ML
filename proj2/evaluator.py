class Evaluator:
    """A class containing loss functions and percent accuracy function to
    evaluate performance of our model"""

    def __init__(self, test_set, predicted_values):
        """Initializes the class with a list of test values and their predicted values"""

        self.test_set = test_set
        self.predicted_values = predicted_values

    def percent_accuracy(self):
        """Returns percent accuracy given true and predicted values"""

        correct = 0
        for i in range(len(self.test_set)):
            if (self.test_set[i].classification == self.predicted_values[i]):
                correct += 1
        return correct / len(self.test_set)

    def one_zero_loss(self):
        """Returns one-zero loss score given true and predicted values"""

        incorrect=0
        for i in range(len(self.test_set)):
            if(self.test_set[i].classification != self.predicted_values[i]):
                incorrect += 1
        return incorrect / len(self.test_set)

    #TODO Add second loss function