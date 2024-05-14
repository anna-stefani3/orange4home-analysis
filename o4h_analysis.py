import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from utils import (
    read_csv_to_dataframe,
    get_activity_duration_data,
    get_cleaned_sensor_dataframe,
    get_motion_count_from_presence_dataframe,
    unique_values_with_count,
    get_balanced_data,
    calculate_metrics,
    print_metrices,
    get_feature_and_label,
    model_train_test_split,
)

from rules import get_location, get_activity_label

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

base_path = "o4h_all_events.csv"

save_path = "o4h_activity_dataframe.csv"

# List of activities
ACTIVITIES_LIST = ["sleeping", "cooking", "bathing", "toileting", "eating", "unknown"]
LOCATIONS_LIST = ["bedroom", "bathroom", "kitchen", "livingroom", "unknown"]


"""
#####################

1 Minute Interval Dataset

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

merged_df.to_csv(save_path)

# Dictionary mapping
activity_mapping = {
    "Using_the_toilet": "toileting",
    "Napping": "sleeping",
    "Showering": "bathing",
    "Using_the_sink": "bathing",
    "Preparing": "cooking",
    "Eating": "eating",
    "Cooking": "cooking",
    "Washing_the_dishes": "cooking",
}

# Update values in 'activity' based on dictionary mapping
merged_df["activity"] = merged_df["activity"].map(lambda x: activity_mapping.get(x, "unknown"))


print("Activity Count before Over Sampling")
print(unique_values_with_count(merged_df, "activity"))


"""
#####################

Balanced Data

#####################
"""
# Reset index to integer-based index instead of Time-based index
resampled_df = merged_df.sample(frac=1.0, random_state=42)
resampled_df.reset_index(drop=True, inplace=True)

print("\n\nShape of final Preprocessed Dataset\nRows =", resampled_df.shape[0], " Columns =", resampled_df.shape[1])

# applying oversampling and getting features and labels
X_resampled, y_resampled = get_feature_and_label(resampled_df)
X_balanced_df, y_balanced_df = get_balanced_data(X_resampled, y_resampled)

print("\n\nShape of Balanced Dataset\nRows =", X_balanced_df.shape[0], " Columns =", X_balanced_df.shape[1])

"""
#####################

UnBalanced Data

#####################
"""
X, y = get_feature_and_label(merged_df)
print("\n\nUnbalanced Data - Features Shape\nRows =", X.shape[0], " Columns =", X.shape[1])
"""
#####################

DECISION TREE

#####################
"""


"""
    ##########################################
    DECISION TREE - Balanced Data
    ##########################################
"""
# parameters for the DecisionTreeClassifier
dt_params = {
    "criterion": "gini",  # 'gini', 'entropy'
    "max_depth": 3,  # 3, 4, 5
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",  # 'sqrt', 'log2'
    "random_state": 42,
}

# Define the DecisionTreeClassifier with the best parameters
decision_tree = DecisionTreeClassifier(**dt_params)

# Train the Decision Tree classifier
model, evaluation_results = model_train_test_split(decision_tree, X_balanced_df, y_balanced_df, ACTIVITIES_LIST)

# Print the evaluation results
print("\n\n")
print("DECISION TREE - (Balanced) METRICES")
print_metrices(evaluation_results)

"""
    ##########################################
    DECISION TREE - UnBalanced Data
    ##########################################
"""

# parameters for the DecisionTreeClassifier
dt_params = {
    "criterion": "gini",  # 'gini', 'entropy'
    "max_depth": 3,  # 3, 4, 5
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",  # 'sqrt', 'log2'
    "random_state": 42,
}

# Define the DecisionTreeClassifier with the best parameters
decision_tree = DecisionTreeClassifier(**dt_params)

# Train the Decision Tree classifier
model, evaluation_results = model_train_test_split(decision_tree, X, y, ACTIVITIES_LIST)

# Print the evaluation results
print("\n\n")
print("DECISION TREE (UnBalanced) METRICES")
print_metrices(evaluation_results)


"""
#####################

RULE BASED SYSTEM

#####################
"""


"""
    ##########################################
    STAGE 1 - LOCATION
    ##########################################
"""

merged_df["location_prediction"] = merged_df.apply(get_location, axis=1).ffill()

# location mapping
location_mapping = {"Bathroom": "bathroom", "Living_room": "livingroom", "Kitchen": "kitchen", "Bedroom": "bedroom"}

# Cleaning the location names
merged_df["location_cleaned"] = merged_df["location"].map(lambda x: location_mapping.get(x, "unknown"))

# Get evaluation_results for location_prediction
evaluation_results = calculate_metrics(merged_df["location_cleaned"], merged_df["location_prediction"], LOCATIONS_LIST)

# Print the evaluation results
print("\n\n")
print("Rule based - Location Classification")
print_metrices(evaluation_results)

"""
    ##########################################
    STAGE 2 - ACTIVITY
    ##########################################
"""
# Storing Activities based on Rules
activities = []
for index in range(merged_df.shape[0]):
    activity = get_activity_label(merged_df, index)
    activities.append(activity)

# Adding activity_prediction into merged_df
merged_df["activity_prediction"] = activities

# getting evaluation_results
evaluation_results = calculate_metrics(merged_df["activity"], merged_df["activity_prediction"], ACTIVITIES_LIST)

# printing evaluation_results
print("\n\n")
print("Rule based - Activity Classification")
print_metrices(evaluation_results)
