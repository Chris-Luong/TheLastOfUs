from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import numpy as np
from grid_search_CV import grid_search_cv
from sklearn.preprocessing import OneHotEncoder

param_space = {
    "n_estimators": [1, 200],
    "learning_rate": [0.05, 20],
}

model = AdaBoostClassifier(estimator=DecisionTreeClassifier(
    max_depth=1), n_estimators=50, learning_rate=1, random_state=42)


class AdaBoost:

    def __init__(self, datasets):
        self.datasets = datasets

    def adaBoost(self, depth, n, a):

        # print(y_encoded)
        # params = grid_search_cv(self.datasets.X_train,
        # self.datasets.y_train.values.ravel(), param_space, model)
        # params = grid_search_cv(
        # self.datasets.X_train, y_encoded, param_space, model)
        #ada = AdaBoostClassifier(**params)
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
            max_depth=1), n_estimators=50, learning_rate=1, random_state=42)

        ada.fit(self.datasets.X_train,
                self.datasets.y_train.to_numpy().flatten())

        print("hmm")
        return ada.predict(self.datasets.X_test)
