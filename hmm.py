import numpy as np
import pandas as pd

# from collections import defaultdict, Counter
# from common_variables import ACTIVITIES_LIST


# def calculate_prior_probabilities(series, observation_to_state):
#     # Convert observations to corresponding states using observation_to_state dict
#     series_states = np.array([observation_to_state[obs] for obs in series])
#     num_states = len(np.unique(series_states))
#     prior_probabilities = np.zeros(num_states)

#     # Count occurrences of each state
#     for state in series_states:
#         prior_probabilities[state] += 1

#     # Normalize to get probabilities
#     prior_probabilities /= len(series_states)

#     return prior_probabilities


# def calculate_transition_probabilities(series, observation_to_state):
#     # Convert observations to corresponding states using observation_to_state dict
#     series_states = np.array([observation_to_state[obs] for obs in series])
#     num_states = len(np.unique(series_states))
#     transitions = np.zeros((num_states, num_states))

#     # Count transitions
#     for i in range(len(series_states) - 1):
#         current_state = series_states[i]
#         next_state = series_states[i + 1]
#         transitions[current_state, next_state] += 1

#     # Normalize to get probabilities
#     row_sums = transitions.sum(axis=1)
#     transition_probabilities = transitions / row_sums[:, np.newaxis]

#     return transition_probabilities


# def calculate_emission_probabilities(series, states):
#     states = np.arange(np.unique(states))
#     num_states = len(np.unique(states))
#     num_observation_values = len(np.unique(series))
#     emission_probabilities = np.zeros((num_states, num_observation_values))

#     # Count occurrences of each observation for each state
#     for i, state in enumerate(states):
#         observation = series[i]
#         emission_probabilities[state, observation] += 1

#     # Normalize to get probabilities
#     row_sums = emission_probabilities.sum(axis=1)
#     emission_probabilities /= row_sums[:, np.newaxis]

#     return emission_probabilities


# def initialize_hmm_probabilities(observed_sequence):
#     # Convert observed_sequence to numpy array
#     observed_sequence = np.array(observed_sequence)

#     # Unique observations
#     unique_observations = ACTIVITIES_LIST
#     num_states = len(unique_observations)
#     num_observations = num_states

#     # Create a mapping from observation to state
#     state_to_observation = {idx: obs for idx, obs in enumerate(unique_observations)}
#     observation_to_state = {obs: idx for idx, obs in enumerate(unique_observations)}

#     # Initial Probabilities (pi)
#     prior_probabilities = calculate_prior_probabilities(observed_sequence, observation_to_state)

#     # Transition Probabilities (A)
#     transition_probabilities = calculate_transition_probabilities(observed_sequence, observation_to_state)

#     # Emission Probabilities (B)
#     emission_probabilities = np.random.dirichlet(np.ones(num_observations), size=num_states)

#     return (
#         prior_probabilities,
#         transition_probabilities,
#         emission_probabilities,
#         state_to_observation,
#     )


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
    # (
    #     prior_probabilities,
    #     transition_probabilities,
    #     emission_probabilities,
    #     state_to_observation,
    # ) = initialize_hmm_probabilities(y_train)

    # print(state_to_observation)
    # model.startprob_ = prior_probabilities
    # model.transmat_ = transition_probabilities
    # model.emissionprob_ = emission_probabilities
    # lengths_train = y_train.value_counts().to_list()

    model.fit(X_train.to_numpy())
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
