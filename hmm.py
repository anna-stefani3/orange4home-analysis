import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from common_variables import ACTIVITIES_LIST


def calculate_prior_probabilities(activity_series):
    # Get unique activities
    unique_activities = activity_series.unique()

    # Calculate the count of the first activity in each sequence
    activity_counts = activity_series.value_counts().reindex(unique_activities, fill_value=0)

    # Apply Laplace smoothing
    alpha = 1
    total_count = activity_counts.sum() + alpha * len(unique_activities)

    # Calculate prior probabilities with smoothing
    prior_probabilities = (activity_counts + alpha) / total_count

    return prior_probabilities.reindex(unique_activities, fill_value=0)


def calculate_transition_probabilities(activity_series):
    # Get unique activities
    unique_activities = activity_series.unique()
    n_activities = len(unique_activities)

    # Create a mapping from activity to index
    activity_to_index = {activity: index for index, activity in enumerate(unique_activities)}

    # Initialize transition matrix with Laplace smoothing
    alpha = 1
    transition_matrix = np.ones((n_activities, n_activities)) * alpha

    # Populate the transition matrix
    for prev_activity, next_activity in zip(activity_series[:-1], activity_series[1:]):
        i = activity_to_index[prev_activity]
        j = activity_to_index[next_activity]
        transition_matrix[i, j] += 1

    # Normalize the transition matrix to get probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_probabilities = transition_matrix / row_sums[:, None]

    return pd.DataFrame(transition_probabilities, index=unique_activities, columns=unique_activities)


def calculate_emission_probabilities(series, states):
    states = np.arange(np.unique(states))
    num_states = len(np.unique(states))
    num_observation_values = len(np.unique(series))
    emission_probabilities = np.zeros((num_states, num_observation_values))

    # Count occurrences of each observation for each state
    for i, state in enumerate(states):
        observation = series[i]
        emission_probabilities[state, observation] += 1

    # Normalize to get probabilities
    row_sums = emission_probabilities.sum(axis=1)
    emission_probabilities /= row_sums[:, np.newaxis]

    return emission_probabilities


def initialize_hmm_probabilities(observed_sequence):
    # Convert observed_sequence to numpy array
    # observed_sequence = np.array(observed_sequence)

    # Unique observations
    unique_observations = ACTIVITIES_LIST
    num_states = len(unique_observations)
    num_observations = num_states

    # Create a mapping from observation to state
    state_to_observation = {idx: obs for idx, obs in enumerate(unique_observations)}
    observation_to_state = {obs: idx for idx, obs in enumerate(unique_observations)}

    # Initial Probabilities (pi)
    prior_probabilities = calculate_prior_probabilities(observed_sequence)

    # Transition Probabilities (A)
    transition_probabilities = calculate_transition_probabilities(observed_sequence)

    # Emission Probabilities (B)
    emission_probabilities = np.random.dirichlet(np.ones(num_observations), size=num_states)

    return (
        prior_probabilities,
        transition_probabilities,
        emission_probabilities,
        state_to_observation,
    )


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
    (
        prior_probabilities,
        transition_probabilities,
        emission_probabilities,
        state_to_observation,
    ) = initialize_hmm_probabilities(y_train)

    print(
        prior_probabilities,
        transition_probabilities,
        emission_probabilities,
        state_to_observation,
    )
    model.startprob_ = prior_probabilities
    model.transmat_ = transition_probabilities
    # model.emissionprob_ = emission_probabilities
    lengths_train = y_train.value_counts().to_list()

    model.fit(X_train.to_numpy())
    states = model.predict(X_train.to_numpy())
    mapping = get_hmm_state_mapping(np.array(states), y_train.to_numpy())

    # Function to predict activities given new sensor event sequences
    def predict_activities(model, test_data):
        states = model.predict(test_data.to_numpy())
        predicted_activities = get_activity_from_hidden_states(pd.Series(states), state_to_observation)
        return predicted_activities

    # Predict activities for X_test
    predicted_activities = predict_activities(model, X_test)
    return predicted_activities
