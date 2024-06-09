import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder


def hmm_get_activity_classification(model, X_train, X_test, y_train, y_test):
    le = LabelEncoder()
    _ = le.fit_transform(y_train)

    lengths_train = y_train.value_counts().to_list()

    model.fit(X_train.to_numpy(), lengths_train)

    # Function to predict activities given new sensor event sequences
    def predict_activities(model, test_data):
        logprob, states = model.decode(test_data.to_numpy(), algorithm="viterbi")
        predicted_activities = le.inverse_transform(states)
        return predicted_activities

    # Predict activities for X_test
    predicted_activities = predict_activities(model, X_test)
    return predicted_activities
