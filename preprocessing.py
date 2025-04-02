"""
Install required modules using command inside quotes

`pip install pandas numpy scikit-learn imbalanced-learn`
"""

import pandas as pd
import os

from utils import (
    get_activity_duration_data,
    get_cleaned_sensor_dataframe,
    get_motion_count_from_presence_dataframe,
    unique_values_with_count,
    get_balanced_data,
    get_feature_and_label,
    get_day_wise_group_df,
)

from rules import is_increasing
from common_variables import BASE_PATH, SAVE_PATH, LOCATION_INT_MAPPING, LOCATION_MAPPING, ACTIVITY_MAPPING
import warnings
from utils import validate_experiment

# Suppress warnings
warnings.filterwarnings("ignore")


def get_balanced_training_dataset(X_train, y_train):
    """
    Balanced Data
    """
    # Merge X_train and y_train together
    dataset = pd.concat([X_train, y_train], axis=1)

    # getting resampled the training dataset
    resampled_df = dataset.sample(frac=1.0, random_state=42)

    # Reset index to integer-based index instead of Time-based index
    resampled_df.reset_index(drop=True, inplace=True)

    # Display the shape of the final preprocessed dataset
    print("\n\nShape of final Preprocessed Dataset\nRows =", resampled_df.shape[0], " Columns =", resampled_df.shape[1])

    # Apply oversampling and extract features and labels
    X_balanced_train, y_balanced_train = get_feature_and_label(resampled_df)
    X_balanced_train_df, y_balanced_train_df = get_balanced_data(X_balanced_train, y_balanced_train)
    X_balanced_train_df.drop(["minute", "hour", "day_of_week", "time_of_day"], axis=1, inplace=True)

    # Display the shape of the balanced dataset after oversampling
    print(
        "\n\nShape of Balanced Dataset\nRows =",
        X_balanced_train_df.shape[0],
        " Columns =",
        X_balanced_train_df.shape[1],
    )
    return X_balanced_train_df, y_balanced_train_df


def apply_location_and_activity_mapping(dataset):

    # Update values in 'activity' column based on the dictionary mapping
    dataset["activity"] = dataset["activity"].map(lambda x: ACTIVITY_MAPPING.get(x, "unknown"))

    # Map and clean the location names using the defined mapping
    dataset["location"] = dataset["location"].map(lambda x: LOCATION_MAPPING.get(x, "unknown"))

    # Map and clean the location names using the defined mapping
    dataset["location_int"] = dataset["location"].map(lambda x: LOCATION_INT_MAPPING.get(x, 5))

    dataset.drop(columns="label", inplace=True)

    return dataset


def add_literature_based_features(dataset):
    """
    Adding New Features
    """

    # Calculate if kitchen humidity is increasing (True/False)
    dataset["kitchen_humidity_is_increasing"] = (
        dataset["kitchen_humidity"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
    )
    # Calculate if kitchen temperature is increasing (True/False)
    dataset["kitchen_temperature_is_increasing"] = (
        dataset["kitchen_temperature"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
    )

    # Identify if kitchen humidity is high (True/False)
    dataset["kitchen_humidity_high"] = dataset["kitchen_humidity"].apply(lambda x: x > 40)

    # Identify if bathroom humidity and presence are high (True/False)
    dataset["bathroom_humidity_high"] = dataset["bathroom_humidity"].apply(lambda x: x > 40)
    dataset["bathroom_presence_high"] = dataset["bathroom_presence"].apply(lambda x: x > 10)

    # Calculate if bathroom humidity is increasing (True/False)
    dataset["bathroom_humidity_is_increasing"] = (
        dataset["bathroom_humidity"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
    )
    # Calculate if bathroom temperature is increasing (True/False)
    dataset["bathroom_temperature_is_increasing"] = (
        dataset["bathroom_temperature"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
    )

    # Identify if living room luminosity and presence are high (True/False)
    dataset["livingroom_luminosity_high"] = dataset["livingroom_luminosity"] > 400
    dataset["livingroom_presence_high"] = dataset["livingroom_presence_table"] > 8

    # Calculate if CO2 levels in the bedroom are increasing (True/False)
    dataset["bedroom_CO2_is_increasing"] = (
        dataset["bedroom_CO2"].rolling(10, min_periods=3).apply(is_increasing, raw=True)
    )

    # Drop rows with missing values
    dataset.dropna(inplace=True)

    return dataset


def add_time_based_features(dataset):
    # Extract time features
    dataset["minute"] = dataset.index.minute
    dataset["hour"] = dataset.index.hour
    dataset["day_of_week"] = dataset.index.dayofweek  # Monday as 0
    dataset["time_of_day"] = pd.cut(dataset.index.hour, bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3])

    # Drop rows with missing values
    dataset.dropna(inplace=True)

    return dataset


def get_lagged_sequence_data(dataset):
    """
    Creating Temporal sequence with past data
    """
    # # Define columns to exclude from duplication
    columns_to_exclude = ["activity", "location_int", "location"]

    # Create shifted versions of the DataFrame for the past rows
    df_shifted_1 = dataset.drop(columns_to_exclude, axis=1).shift(1).add_suffix("_t1")
    df_shifted_2 = dataset.drop(columns_to_exclude, axis=1).shift(1).add_suffix("_t2")

    # Concatenate past rows into current row to create combined data point
    time_series_df = pd.concat([df_shifted_2, df_shifted_1, dataset], axis=1).dropna()

    return time_series_df


def get_training_validation_and_testing_split_data(dataset, indexes):
    # get the group ids (For group selection)
    group_ids = list(indexes.index)

    # split point for training data at 50% -> 10 Days of data
    train_split = (len(group_ids) * 2) // 4

    # split point at 10-15 for validation and 16-20 for testing
    validation_split = (train_split + len(group_ids)) // 2

    print(f"Total Number of Days in Dataset : {len(indexes)} and Split Point at {train_split}")
    # getting the training, validation and testing from GROUPED_DAYWISE dataset
    training = get_day_wise_group_df(dataset, group_ids, start_day=0, end_day=train_split)
    validation = get_day_wise_group_df(dataset, group_ids, start_day=train_split, end_day=validation_split)
    testing = get_day_wise_group_df(dataset, group_ids, start_day=validation_split, end_day=len(group_ids))

    # Display activity counts before oversampling
    print("\n\nActivity Count after common preprocessing")
    print(unique_values_with_count(training, "activity"))

    return training, validation, testing


def apply_preprocessing(filename=BASE_PATH):
    if os.path.exists(SAVE_PATH):
        response = input("Press 1 to Read from saved Preprocessed CSV File")
        read_saved_file = True if response == "1" else False
    if os.path.exists(SAVE_PATH) and read_saved_file:
        # Reading from preprocessed CSV File if Exists
        merged_df = pd.read_csv(SAVE_PATH, index_col="Time", parse_dates=["Time"])
    else:
        # Read CSV file into a DataFrame
        df = pd.read_csv(filename)

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
        merged_df.to_csv(SAVE_PATH)

    """
    Daywise Grouping and then splitting dataset into training, validation and testing
    """

    # grouping data daywise
    GROUPED_DAYWISE = merged_df.groupby(pd.Grouper(freq="D"))

    # getting number of rows in each group data
    groups_size = GROUPED_DAYWISE.size()

    # filtering groups which has more than 0 rows
    non_empty_groups = groups_size[groups_size > 0]

    training, validation, testing = get_training_validation_and_testing_split_data(
        dataset=GROUPED_DAYWISE, indexes=non_empty_groups
    )

    training = apply_location_and_activity_mapping(training)
    validation = apply_location_and_activity_mapping(validation)
    testing = apply_location_and_activity_mapping(testing)

    response = input("Press 1 to Add Literature Based Features\n")
    if response == "1":
        training = add_literature_based_features(training)
        validation = add_literature_based_features(validation)
        testing = add_literature_based_features(testing)
        training = add_time_based_features(training)
        validation = add_time_based_features(validation)
        testing = add_time_based_features(testing)

    response = input("Press 1 to Add Lagged Features\n")
    if response == "1":
        training = get_lagged_sequence_data(training)
        validation = get_lagged_sequence_data(validation)
        testing = get_lagged_sequence_data(testing)

    # Extract features and labels from the training, validation and testing DataFrame
    X_train, y_train = get_feature_and_label(training)
    X_valid, y_valid = get_feature_and_label(validation)
    X_test, y_test = get_feature_and_label(testing)

    # Display the shape of the unbalanced data features
    print("\n\nUnbalanced Data - Features Shape\nRows =", X_train.shape[0], " Columns =", X_train.shape[1])

    return X_train, X_valid, X_test, y_train, y_valid, y_test
