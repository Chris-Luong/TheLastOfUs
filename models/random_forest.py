from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def RF(data):
    # Initialize a DecisionTreeClassifier object
    clf = RandomForestClassifier(n_estimators=100)

    # Fit the classifier to the training data
    clf.fit(data.X_train, data.y_train)

    # Use the trained classifier to make predictions on the test data
    y_pred = clf.predict(data.X_test)

    # Calculate the accuracy of the classifier on the test data
    accuracy = accuracy_score(data.y_test, y_pred)

    # Print the accuracy
    print(f"Accuracy: {accuracy}")

    return y_pred