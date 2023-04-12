from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score

# These parameters based off grid search with cross validation
# params = {
#     # "n_estimators": 80,
#     "min_samples_leaf": 30,
#     "min_samples_split": 2,
#     "max_depth": 20,
#     "max_features": 60,
#     'criterion': "gini",
#     'n_jobs': 15,
#     'random_state': 42
# }
params = {
    'n_estimators': 80,
    'max_depth': 20,
    'max_features': 60,
    'min_samples_leaf': 31,
    'min_samples_split': 2,
    'criterion': "gini",
    'n_jobs': 15,
    'random_state': 42,
}
def grid_search_cv(x_train, y_train):
    # The parameter space
    # parameter_space = {
    #     "min_samples_leaf": [30, 31],
    #     "min_samples_split": [2, 10],
    #     "max_depth": [18, 20],
    #     "max_features": [60, 80]
    # }
    # Takes very long
    parameter_space = {
        "min_samples_leaf": [30, 31, 40],
        "min_samples_split": [2, 3, 10],
        "max_depth": [9, 10, 18, 20],
        "max_features": [60, 80, 100]
    }
    # do the grid search with cross validation
    print("Tuning hyper-parameters by gridsearch")
    model = RandomForestClassifier(
        criterion="gini",
        n_jobs=15,
        random_state=42)
    grid = GridSearchCV(model, parameter_space, cv=5) # scoring="accuracy" ?
    grid.fit(x_train, y_train)

    # get the best parameters on the sample of training set
    print("The best parameters are: ", grid.best_params_)
    print("Best score is: ", grid.best_score_)
    return grid.best_params_

def RF(data):
    # params = grid_search_cv(data.X_train, data.y_train)
    # Initialize a DecisionTreeClassifier object
    clf = RandomForestClassifier(**params)

    # Fit the classifier to the training data
    clf.fit(data.X_train, data.y_train)

    # Use the trained classifier to make predictions on the test data
    y_pred = clf.predict(data.X_test)

    # # Calculate the accuracy of the classifier on the test data
    # accuracy = accuracy_score(data.y_test, y_pred)

    # # Print the accuracy
    # print(f"Accuracy: {accuracy}")

    return y_pred