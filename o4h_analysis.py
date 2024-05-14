"""
Install required modules using command inside quotes

`pip install pandas numpy scikit-learn imbalanced-learn`
"""
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

# Suppress warnings
warnings.filterwarnings("ignore")

# Define file paths
base_path = "o4h_all_events.csv"
save_path = "o4h_activity_dataframe.csv"

# lists for activities and locations
ACTIVITIES_LIST = ["sleeping", "cooking", "bathing", "toileting", "eating", "unknown"]
LOCATIONS_LIST = ["bedroom", "bathroom", "kitchen", "livingroom", "unknown"]

"""
#####################

1 Minute Interval Dataset

#####################
"""
# Read CSV file into a DataFrame
df = read_csv_to_dataframe(base_path)

# Extract activity duration data
get_activity_duration_data(df)

# Clean and prepare the sensor DataFrame
sensor_df = get_cleaned_sensor_dataframe(df, frequency="1T")

# Calculate motion count from the presence DataFrame
motion_count_df = get_motion_count_from_presence_dataframe(df, frequency="1T")

# Merge the sensor DataFrame and motion count DataFrame based on the 'Time' column
merged_df = pd.merge(sensor_df, motion_count_df, on="Time", how="inner")

# Identify columns that can be converted to float type
float_columns = merged_df.columns.difference(["location", "activity", "label"])

# Convert selected columns to float
merged_df[float_columns] = merged_df[float_columns].astype(float)

# Save the merged DataFrame to a CSV file
merged_df.to_csv(save_path)

# Define a dictionary for mapping activities
activity_mapping = {
    "Using_the_toilet": "toileting",
    "Napping": "sleeping",
    "Showering": "bathing",
    "Using_the_sink": "toileting",
    "Preparing": "cooking",
    "Eating": "eating",
    "Cooking": "cooking",
    "Washing_the_dishes": "cooking",
}

# Update values in 'activity' column based on the dictionary mapping
merged_df["activity"] = merged_df["activity"].map(lambda x: activity_mapping.get(x, "unknown"))

# Display activity counts before oversampling
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

# Display the shape of the final preprocessed dataset
print("\n\nShape of final Preprocessed Dataset\nRows =", resampled_df.shape[0], " Columns =", resampled_df.shape[1])

# Apply oversampling and extract features and labels
X_resampled, y_resampled = get_feature_and_label(resampled_df)
X_balanced_df, y_balanced_df = get_balanced_data(X_resampled, y_resampled)

# Display the shape of the balanced dataset after oversampling
print("\n\nShape of Balanced Dataset\nRows =", X_balanced_df.shape[0], " Columns =", X_balanced_df.shape[1])

"""
#####################

UnBalanced Data

#####################
"""
# Extract features and labels from the merged DataFrame
X, y = get_feature_and_label(merged_df)

# Display the shape of the unbalanced data features
print("\n\nUnbalanced Data - Features Shape\nRows =", X.shape[0], " Columns =", X.shape[1])

"""
#####################

DECISION TREE

#####################
"""

"""
    ##########################################
    DECISION TREE - Initialising
    ##########################################
"""

# Define parameters for the DecisionTreeClassifier
dt_params = {
    "criterion": "gini",  # Splitting criterion: 'gini' or 'entropy'
    "max_depth": 5,  # Maximum depth of the decision tree
    "min_samples_split": 2,  # Minimum samples required to split an internal node
    "min_samples_leaf": 1,  # Minimum samples required to be at a leaf node
    "max_features": "sqrt",  # Number of features to consider for the best split
    "random_state": 42,  # Random state for reproducibility
}

# Initialize the DecisionTreeClassifier with the specified parameters
decision_tree = DecisionTreeClassifier(**dt_params)


"""
    ##########################################
    DECISION TREE - Balanced Data
    ##########################################
"""
# Train the Decision Tree classifier using the balanced dataset
model, evaluation_results = model_train_test_split(decision_tree, X_balanced_df, y_balanced_df, ACTIVITIES_LIST)

# Display the evaluation results for the trained Decision Tree model
print("\n\n")
print("DECISION TREE - (Balanced) METRICS")
print_metrices(evaluation_results)


"""
    ##########################################
    DECISION TREE - UnBalanced Data
    ##########################################
"""

# Train the Decision Tree classifier using the unbalanced dataset
model, evaluation_results = model_train_test_split(decision_tree, X, y, ACTIVITIES_LIST)

# Display the evaluation results for the trained Decision Tree model
print("\n\n")
print("DECISION TREE (Unbalanced) METRICS")
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
# Apply the get_location function to each row to predict the location and forward fill any missing values
merged_df["location_prediction"] = merged_df.apply(get_location, axis=1).ffill()

# Define a dictionary for mapping the location names
location_mapping = {"Bathroom": "bathroom", "Living_room": "livingroom", "Kitchen": "kitchen", "Bedroom": "bedroom"}

# Map and clean the location names using the defined mapping
merged_df["location_cleaned"] = merged_df["location"].map(lambda x: location_mapping.get(x, "unknown"))

# Calculate evaluation metrics for the predicted location against the actual cleaned location
evaluation_results = calculate_metrics(merged_df["location_cleaned"], merged_df["location_prediction"], LOCATIONS_LIST)

# Print the evaluation results for the location prediction
print("\n\n")
print("Rule-based - Location Classification")
print_metrices(evaluation_results)


"""
    ##########################################
    STAGE 2 - ACTIVITY
    ##########################################
"""
# Initialize an empty list to store predicted activities based on rules
activities = []

# Loop through each row in the merged DataFrame to predict the activity
for index in range(merged_df.shape[0]):
    activity = get_activity_label(merged_df, index)
    activities.append(activity)

# Add the predicted activities to the merged DataFrame
merged_df["activity_prediction"] = activities

# Calculate evaluation metrics for the predicted activities against the actual activities
evaluation_results = calculate_metrics(merged_df["activity"], merged_df["activity_prediction"], ACTIVITIES_LIST)

# Print the evaluation results for the activity prediction
print("\n\n")
print("Rule-based - Activity Classification")
print_metrices(evaluation_results)
