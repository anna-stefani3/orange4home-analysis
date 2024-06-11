import numpy as np
import pandas as pd


def get_hmm_state_mapping(states, labels):
    # Initialize an empty dictionary to store the mapping
    mapping = {}
    for state in np.unique(states):
        # Filter the DataFrame to include only rows with the current hidden state
        subset = states == state
        # Find the most common location associated with the current hidden state

        # Get unique values and their counts
        values, counts = np.unique(labels[subset], return_counts=True)

        # Find the index of the maximum count
        max_count_index = np.argmax(counts)

        # The most common value
        most_common_value = values[max_count_index]

        # most_common_value = np.bincount(subset).argmax()
        # Add the mapping between the hidden state and its most common location to the dictionary
        mapping[state] = most_common_value

    # Print the mapping dictionary
    print("HMM State to Label Mapping :", mapping)

    # Add a new column "hidden_label" to the DataFrame by mapping each hidden state to its most common location
    return mapping


def get_activity_from_hidden_states(states, mapping):
    return states.map(mapping)


def hmm_get_activity_classification(model, X_train, X_test, y_train, y_test):
    lengths_train = y_train.value_counts().to_list()

    model.fit(X_train.to_numpy(), lengths_train)
    states = model.predict(X_train.to_numpy())
    mapping = get_hmm_state_mapping(np.array(states), y_train.to_numpy())

    # Function to predict activities given new sensor event sequences
    def predict_activities(model, test_data):
        states = model.predict(test_data.to_numpy())
        predicted_activities = get_activity_from_hidden_states(pd.Series(states), mapping)
        return predicted_activities

    # Predict activities for X_test
    predicted_activities = predict_activities(model, X_test)
    return predicted_activities
