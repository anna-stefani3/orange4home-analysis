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
    get_decision_tree_structure,
)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

base_path = "o4h_all_events.csv"


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

print("Shape of final Preprocessed Dataset\nRow =", merged_df.shape[0], " Columns =", merged_df.shape[1])

# applying oversampling and getting features and labels
X_resampled_df, y_resampled_df = get_balanced_data(merged_df)

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
print("DECISION TREE METRICES - 1 Minute Interval Dataset")
print(evaluation_results)

# Assuming you have trained a decision tree model named `decision_tree`
# Get the decision tree structure as text
tree_structure = get_decision_tree_structure(model, X_resampled_df.columns.to_list())

# Print the decision tree structure
print("Decision Tree Structure:\n")
print(tree_structure)

"""
#####################

10 Minute Interval

#####################
"""

# Read CSV file into a DataFrame
df = read_csv_to_dataframe(base_path)

get_activity_duration_data(df)

# Clean and prepare the sensor DataFrame
sensor_df = get_cleaned_sensor_dataframe(df, frequency="10T")

# Calculate motion count from the presence DataFrame
motion_count_df = get_motion_count_from_presence_dataframe(df, frequency="10T")

# Merge the sensor DataFrame and motion count DataFrame based on the 'Time' column
merged_df = pd.merge(sensor_df, motion_count_df, on="Time", how="inner")

# getting columns which can be converted to float type
float_columns = merged_df.columns.difference(["location", "activity", "label"])

# Convert selected columns to float
merged_df[float_columns] = merged_df[float_columns].astype(float)

# List of activities to filter from dataset
selected_activities = ["Using_the_toilet", "Napping", "Showering", "Eating", "Cooking"]

# selecting only required activities from dataset
merged_df = filter_rows_based_on_given_values_list(merged_df, selected_activities, column="activity")

print("Activity Count before Over Sampling")
print(unique_values_with_count(merged_df, "activity"))

# Reset index to integer-based index instead of Time-based index
merged_df = merged_df.sample(frac=1.0, random_state=42)
merged_df.reset_index(drop=True, inplace=True)

print("Shape of final Preprocessed Dataset\nRow =", merged_df.shape[0], " Columns =", merged_df.shape[1])

# applying oversampling and getting features and labels
X_resampled_df, y_resampled_df = get_balanced_data(merged_df)

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
print("DECISION TREE METRICES - 10 Minute Interval Dataset")
print(evaluation_results)

# Assuming you have trained a decision tree model named `decision_tree`
# Get the decision tree structure as text
tree_structure = get_decision_tree_structure(model, X_resampled_df.columns.to_list())

# Print the decision tree structure
print("Decision Tree Structure:\n")
print(tree_structure)
