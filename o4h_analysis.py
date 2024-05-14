"""
Install required modules using command inside quotes

`pip install pandas numpy scikit-learn hmmlearn imbalanced-learn`
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from utils import (
    read_csv_to_dataframe,
    get_activity_duration_data,
    get_cleaned_sensor_dataframe,
    get_motion_count_from_presence_dataframe,
    filter_rows_based_on_given_values_list,
    unique_values_with_count,
    get_balanced_data,
    model_training_and_testing,
    calculate_metrics,
)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

base_path = "o4h_all_events.csv"

# List of activities to filter from dataset
SELECTED_ACTIVITIES = ["Using_the_toilet", "Napping", "Showering", "Eating", "Cooking"]


"""
#####################

1 Minute Interval

#####################
"""

# Read CSV file into a DataFrame
df = read_csv_to_dataframe(base_path)

get_activity_duration_data(df)

# Clean and prepare the sensor DataFrame
sensor_df = get_cleaned_sensor_dataframe(df, frequency="1T")

# Calculate motion count from the presence DataFrame
motion_count_df = get_motion_count_from_presence_dataframe(df, frequency="1T")

# Merge the sensor DataFrame and motion count DataFrame based on the 'Time' column
merged_df = pd.merge(sensor_df, motion_count_df, on="Time", how="inner")

# getting columns which can be converted to float type
float_columns = merged_df.columns.difference(["location", "activity", "label"])

# Convert selected columns to float
merged_df[float_columns] = merged_df[float_columns].astype(float)

# List of activities to filter from dataset
selected_activities = ["Using_the_toilet", "Napping", "Showering", "Eating", "Cooking"]
print("SELECTED ACTIVITES -> ", selected_activities)

# selecting only required activities from dataset
merged_df = filter_rows_based_on_given_values_list(merged_df, selected_activities, column="activity")

print("Activity Count before Over Sampling")
print(unique_values_with_count(merged_df, "activity"))

# Reset index to integer-based index instead of Time-based index
merged_df = merged_df.sample(frac=1.0, random_state=42)
merged_df.reset_index(drop=True, inplace=True)

print("\n\nShape of final Preprocessed Dataset\nRow =", merged_df.shape[0], " Columns =", merged_df.shape[1])

# applying oversampling and getting features and labels
X_resampled_df, y_resampled_df = get_balanced_data(merged_df)


"""
#####################

DECISION TREE

#####################
"""


# parameters for the DecisionTreeClassifier
dt_params = {
    "criterion": "gini",  # 'gini', 'entropy'
    "max_depth": 7,  # 5, 6, 7, 8, 9, 10
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",  # 'sqrt', 'log2'
    "random_state": 42,
}

# Define the DecisionTreeClassifier with the best parameters
decision_tree = DecisionTreeClassifier(**dt_params)

# Train the Decision Tree classifier on the resampled data using 10-fold cross-validation
model, evaluation_results = model_training_and_testing(decision_tree, X_resampled_df, y_resampled_df)

# Print the evaluation results
print("DECISION TREE METRICES")
print(evaluation_results)

print("\n\n\n")
"""
#####################

SVM MODEL

#####################
"""
from sklearn.svm import SVC

svm_params = {"kernel": "linear", "C": 1.0}

# Create SVM classifier
svm_classifier = SVC(**svm_params)

# Train the Decision Tree classifier on the resampled data using 10-fold cross-validation
model, evaluation_results = model_training_and_testing(svm_classifier, X_resampled_df, y_resampled_df)

# Print the evaluation results
print("SVM METRICES")
print(evaluation_results)
print("\n\n\n")
"""
#####################

RANDOM FOREST

#####################
"""
from sklearn.ensemble import RandomForestClassifier

"""
Param Explaination
- n_estimators: The number of trees in the forest (default: 100)
- max_depth: The maximum depth of the trees (default: None)
- min_samples_split: The minimum number of samples required to split an internal node (default: 2)
- min_samples_leaf: The minimum number of samples required to be at a leaf node (default: 1)
- random_state: Random seed for reproducibility (default: None)
"""

rf_params = {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42}

# Define the RandomForestClassifier with the best parameters
rf_classifier = RandomForestClassifier(**rf_params)

# Train the RandomForestClassifier on the resampled data using 10-fold cross-validation
model, evaluation_results = model_training_and_testing(rf_classifier, X_resampled_df, y_resampled_df)

# Print the evaluation results
print("RANDOM FOREST METRICES")
print(evaluation_results)

print("\n\n\n")

"""
#####################

HMM MODEL

#####################
"""
from hmmlearn import hmm

# Define the number of hidden states for the HMM
num_hidden_states = 8

# Initialize a Gaussian HMM model with specified parameters
model = hmm.GaussianHMM(n_components=num_hidden_states, covariance_type="full", n_iter=100)


# Drop columns that are not used as features for the model
X = X_resampled_df

# Convert the data in X to numeric format, coercing errors to NaN
X = X.apply(pd.to_numeric, errors="coerce")

# Fit the HMM model to the data
model.fit(X)

# Predict the hidden states for each observation in X
hidden_states = model.predict(X)

hmm_df = pd.DataFrame()

# Add the predicted hidden states as a new column in the DataFrame
hmm_df["hidden_state"] = hidden_states
hmm_df["activity"] = y_resampled_df["activity"]


# Initialize an empty dictionary to store the mapping
mapping = {}

# Iterate over each unique hidden state in the HMM Model
for state in hmm_df["hidden_state"].unique():
    # Filter the DataFrame to include only rows with the current hidden state
    subset = hmm_df[hmm_df["hidden_state"] == state]

    # Find the most common activity associated with the current hidden state
    most_common_activity = subset["activity"].mode().iloc[0]
    # Add the mapping between the hidden state and its most common activity to the dictionary
    mapping[state] = most_common_activity

# Print the mapping dictionary
print("HMM [State:Label] Mapping :", mapping)


# Add a new column "hidden_label" to the DataFrame by mapping each hidden state to its most common activity
hmm_df["hidden_label"] = hmm_df["hidden_state"].map(mapping)
hmm_metric_scores = calculate_metrics(hmm_df["activity"], hmm_df["hidden_label"], SELECTED_ACTIVITIES)
print("HMM Model Metrices")
print(hmm_metric_scores)
