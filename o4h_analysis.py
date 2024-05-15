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
    model_train_test_score,
    get_day_wise_group_df,
    get_top_features_using_RFECV,
    get_top_features_using_RFE,
)

from rules import get_location, get_activity_label, is_increasing

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

# Define a dictionary for mapping the location names
location_mapping = {"Bathroom": "bathroom", "Living_room": "livingroom", "Kitchen": "kitchen", "Bedroom": "bedroom"}

# Define a dictionary for mapping the location to int (Needed For Decision Tree)
location_int_mapping = {"bathroom": 1, "livingroom": 2, "kitchen": 3, "bedroom": 4, "unknown": 5}


# Map and clean the location names using the defined mapping
merged_df["location"] = merged_df["location"].map(lambda x: location_mapping.get(x, "unknown"))

# Map and clean the location names using the defined mapping
merged_df["location_int"] = merged_df["location"].map(lambda x: location_int_mapping.get(x, 5))

# removing useless column -> "label"
merged_df.drop(columns="label", inplace=True)


"""
#####################

Adding New Features

#####################
"""

# Calculate if kitchen humidity is increasing (True/False)
merged_df["kitchen_humidity_is_increasing"] = (
    merged_df["kitchen_humidity"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
)
# Calculate if kitchen temperature is increasing (True/False)
merged_df["kitchen_temperature_is_increasing"] = (
    merged_df["kitchen_temperature"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
)

# Identify if kitchen humidity is high (True/False)
merged_df["kitchen_humidity_high"] = merged_df["kitchen_humidity"].apply(lambda x: x > 40)

# Identify if bathroom humidity and presence are high (True/False)
merged_df["bathroom_humidity_high"] = merged_df["bathroom_humidity"].apply(lambda x: x > 40)
merged_df["bathroom_presence_high"] = merged_df["bathroom_presence"].apply(lambda x: x > 10)

# Calculate if bathroom humidity is increasing (True/False)
merged_df["bathroom_humidity_is_increasing"] = (
    merged_df["bathroom_humidity"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
)
# Calculate if bathroom temperature is increasing (True/False)
merged_df["bathroom_temperature_is_increasing"] = (
    merged_df["bathroom_temperature"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
)

# Identify if living room luminosity and presence are high (True/False)
merged_df["livingroom_luminosity_high"] = merged_df["livingroom_luminosity"] > 400
merged_df["livingroom_presence_high"] = merged_df["livingroom_presence_table"] > 8

# Calculate if CO2 levels in the bedroom are increasing (True/False)
merged_df["bedroom_CO2_is_increasing"] = (
    merged_df["bedroom_CO2"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
)

# Extract time features
merged_df["minute"] = merged_df.index.minute
merged_df["hour"] = merged_df.index.hour
merged_df["day_of_week"] = merged_df.index.dayofweek  # Monday as 0
merged_df["time_of_day"] = pd.cut(merged_df.index.hour, bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3])

# Drop rows with missing values
merged_df.dropna(inplace=True)


"""
#####################

Creating Temporal sequence with past data

#####################
"""

# Define columns to exclude from duplication
columns_to_exclude = ["activity", "location_int", "location"]

# Create shifted versions of the DataFrame for the past rows
df_shifted_1 = merged_df.drop(columns_to_exclude, axis=1).shift(1).add_suffix("_t1")
df_shifted_2 = merged_df.drop(columns_to_exclude, axis=1).shift(2).add_suffix("_t2")
df_shifted_3 = merged_df.drop(columns_to_exclude, axis=1).shift(3).add_suffix("_t3")
df_shifted_4 = merged_df.drop(columns_to_exclude, axis=1).shift(4).add_suffix("_t4")
df_shifted_5 = merged_df.drop(columns_to_exclude, axis=1).shift(5).add_suffix("_t5")

# Concatenate past rows into current row to create combined data point
time_series_df = pd.concat(
    [df_shifted_5, df_shifted_4, df_shifted_3, df_shifted_2, df_shifted_1, merged_df], axis=1
).dropna()


"""
#####################

Daywise Data Grouping

#####################
"""

"""
#####################

Daywise Data Grouping

#####################
"""

# grouping data daywise
GROUPED_DAYWISE = time_series_df.groupby(pd.Grouper(freq="D"))

# getting number of rows in each group data
groups_size = GROUPED_DAYWISE.size()

# filtering groups which has more than 0 rows
groups_size = groups_size[groups_size > 0]

# split point at 50% -> 10 Days of data
split_point_for_dataset = (len(groups_size) * 2) // 4

print(f"Total Number of Days in Dataset : {len(groups_size)} and Split Point at {split_point_for_dataset}")

# get the group ids (For group selection)
group_ids = list(groups_size.index)

# getting the training data from GROUPED_DAYWISE dataset
training = get_day_wise_group_df(GROUPED_DAYWISE, group_ids, start_day=0, end_day=split_point_for_dataset)

# getting the testing data from GROUPED_DAYWISE dataset
testing = get_day_wise_group_df(GROUPED_DAYWISE, group_ids, start_day=split_point_for_dataset, end_day=len(groups_size))

# Display activity counts before oversampling
print("\n\nActivity Count before Over Sampling")
print(unique_values_with_count(training, "activity"))


"""
#####################

Balanced Data

#####################
"""
# getting resampled the training dataset
resampled_df = training.sample(frac=1.0, random_state=42)
# Reset index to integer-based index instead of Time-based index
resampled_df.reset_index(drop=True, inplace=True)

# Display the shape of the final preprocessed dataset
print("\n\nShape of final Preprocessed Dataset\nRows =", resampled_df.shape[0], " Columns =", resampled_df.shape[1])

# Apply oversampling and extract features and labels
X_balanced_train, y_balanced_train = get_feature_and_label(resampled_df)
X_balanced_train_df, y_balanced_train_df = get_balanced_data(X_balanced_train, y_balanced_train)

# Display the shape of the balanced dataset after oversampling
print("\n\nShape of Balanced Dataset\nRows =", X_balanced_train_df.shape[0], " Columns =", X_balanced_train_df.shape[1])

"""
#####################

UnBalanced Data

#####################
"""
# Extract features and labels from the training DataFrame
X_train, y_train = get_feature_and_label(training)

X_test, y_test = get_feature_and_label(testing)

# Display the shape of the unbalanced data features
print("\n\nUnbalanced Data - Features Shape\nRows =", X_train.shape[0], " Columns =", X_train.shape[1])


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


"""
    ##########################################
    DECISION TREE - Balanced Data
    ##########################################
"""
# Initialize the DecisionTreeClassifier with the specified parameters
DT = DecisionTreeClassifier(**dt_params)

# Train the Decision Tree classifier using the balanced dataset
model, evaluation_results, _ = model_train_test_score(
    DT, X_balanced_train_df, X_test, y_balanced_train_df["activity"], y_test["activity"], ACTIVITIES_LIST
)

# Display the evaluation results for the trained Decision Tree model
print("\n\n")
print("DECISION TREE - (Balanced) METRICS")
print_metrices(evaluation_results)


"""
    ##########################################
    DECISION TREE - Classify Activity Directly
    ##########################################
"""
print("\n\nMETHOD 1 - Classify Activity Directly")
# Initialize Decision Tree Classifier with specified parameters
DT = DecisionTreeClassifier(**dt_params)

# Get top features based on importance for activity prediction
top_features = get_top_features_using_RFECV(X_train, y_train["activity"])
X_train_activity_subset = X_train[top_features]
X_test_activity_subset = X_test[top_features]

# Train and evaluate the model for activity prediction
model, evaluation_results, _ = model_train_test_score(
    DT, X_train_activity_subset, X_test_activity_subset, y_train["activity"], y_test["activity"], ACTIVITIES_LIST
)

print("DECISION TREE - (Unbalanced) Activity METRICS")
print_metrices(evaluation_results)


"""
    ##########################################
    DECISION TREE - Location then Activity 
    ##########################################
"""
print("\n\nMETHOD 2 - Classify Location then Activity")
# Initialize Decision Tree Classifier with specified parameters
DT = DecisionTreeClassifier(**dt_params)

# Select features related to presence in different locations
X_train_location = X_train[["bedroom_presence", "kitchen_presence", "bathroom_presence", "livingroom_presence_table"]]
X_test_location = X_test[["bedroom_presence", "kitchen_presence", "bathroom_presence", "livingroom_presence_table"]]

# Train and evaluate the model for location prediction
model, evaluation_results, location = model_train_test_score(
    DT, X_train_location, X_test_location, y_train["location"], y_test["location"], LOCATIONS_LIST
)

print("\n\n")
print("DECISION TREE - (Unbalanced) LOCATION METRICS")
print_metrices(evaluation_results)


# Create a copy of the testing data for activity prediction and add location_int column
X_test_activity = X_test.copy()
X_test_activity["location_int"] = [location_int_mapping[key] for key in location]

# Create a copy of the training data for activity prediction and add location_int column
X_train_activity = X_train.copy()
X_train_activity["location_int"] = y_train["location_int"]

# Initialize Decision Tree Classifier with specified parameters
DT = DecisionTreeClassifier(**dt_params)

# Get top features based on importance for activity prediction
top_features = get_top_features_using_RFECV(X_train_activity, y_train["activity"])
X_train_activity_subset = X_train_activity[top_features]
X_test_activity_subset = X_test_activity[top_features]

# Train and evaluate the model for activity prediction
model, evaluation_results, _ = model_train_test_score(
    DT, X_train_activity_subset, X_test_activity_subset, y_train["activity"], y_test["activity"], ACTIVITIES_LIST
)

print("DECISION TREE - (Unbalanced) Activity METRICS")
print_metrices(evaluation_results)


"""
    ##########################################
    DECISION TREE - Each Activity Separately
    ##########################################
"""
print("\n\nMETHOD 3 - Classify Each Activity Separately")
# Create copies of training and testing data
X_train_activity = X_train.copy()
X_test_activity = X_test.copy()

# Mapping activities to locations
activity_location_map = {
    "sleeping": "bedroom",
    "cooking": "kitchen",
    "bathing": "bathroom",
    "toileting": "bathroom",
    "eating": "livingroom",
}

# Iterate over activities
for activity in ACTIVITIES_LIST[:-1]:

    # Filter labels for the current activity
    y_train_activity = y_train["activity"].apply(lambda x: x if x == activity else "unknown")
    y_test_activity = y_test["activity"].apply(lambda x: x if x == activity else "unknown")

    # Select features related to the room of the current activity
    room_features = [col for col in X_train_activity.columns if col.startswith(activity_location_map[activity])]

    # Get top features based on importance
    top_features = get_top_features_using_RFE(X_train_activity[room_features], y_train_activity)
    X_train_activity_subset = X_train_activity[top_features]
    X_test_activity_subset = X_test_activity[top_features]

    # Initialize Decision Tree Classifier
    DT = DecisionTreeClassifier(**dt_params)

    # Train and evaluate the model
    model, evaluation_results, _ = model_train_test_score(
        DT, X_train_activity_subset, X_test_activity_subset, y_train_activity, y_test_activity, [activity, "unknown"]
    )

    # Print evaluation metrics for the current activity
    print("DECISION TREE -> ", activity)
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
location_prediction = X_test.apply(get_location, axis=1).ffill()

# Calculate evaluation metrics for the predicted location against the actual cleaned location
evaluation_results = calculate_metrics(y_test["location"], location_prediction, LOCATIONS_LIST)

# Print the evaluation results for the location prediction
print("\n\n")
print("Rule-based - Location Classification")
print_metrices(evaluation_results)

"""
    ##########################################
    STAGE 2 - ACTIVITY
    ##########################################
"""
X_test_rules = X_test.copy()
X_test_rules["location_prediction"] = location_prediction
# Initialize an empty list to store predicted activities based on rules
activities = []

# Loop through each row in the merged DataFrame to predict the activity
for index in range(10, X_test.shape[0]):
    activity = get_activity_label(X_test_rules, index)
    activities.append(activity)

y_test_rules = y_test.iloc[10:]

# Add the predicted activities to the merged DataFrame
y_test_rules["activity_prediction"] = activities

# Calculate evaluation metrics for the predicted activities against the actual activities
evaluation_results = calculate_metrics(y_test_rules["activity"], y_test_rules["activity_prediction"], ACTIVITIES_LIST)

# Print the evaluation results for the activity prediction
print("\n\n")
print("Rule-based - Activity Classification")
print_metrices(evaluation_results)
