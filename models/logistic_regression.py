# Logistic Regression algorithm

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


class LogReg:

    def __init__(self, datasets):
        self.datasets = datasets

    # TODO: Fix so you can input penalty and C
    def one_v_rest(self):
        """
         - One vs Rest logistic regression
         - Returns:
            - y prediction on X_test
        """
        # Create a one-vs-all logistic regression model and fit it to the training data
        # Saga is recommended for larger datasets and uses a version of SGD
        lr = LogisticRegression(
            max_iter=1000, solver="saga", multi_class='ovr', penalty="l1", C=0.1)
        ova = OneVsRestClassifier(lr)
        ova.fit(self.datasets.X_train, self.datasets.y_train)

        # Make predictions on the testing data
        y_pred = ova.predict(self.datasets.X_test)

        return y_pred

    # Probabilities?
    #
