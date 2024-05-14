"""
Install required modules using command inside quotes

`pip install pandas numpy scikit-learn hmmlearn`
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from hmmlearn import hmm

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Model is not converging*")


base_path = "data/o4h_all_events.csv"


def read_csv_to_dataframe(csv_file):
    """
    Function to read a CSV file into a DataFrame.

    Parameters:
    csv_file (str): The file path or name of the CSV file to be read.

    Returns:
    pandas.DataFrame: The DataFrame containing the data from the CSV file.
    """
    dataframe = pd.read_csv(csv_file)
    return dataframe


def unique_values_in_column(dataframe, column_name):
    """
    Function to return unique values in a specified column of a DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column for which unique values are desired.

    Returns:
    list: A list containing the unique values present in the specified column.
    """
    unique_values = dataframe[column_name].unique().tolist()
    return unique_values


def unique_values_with_count(dataframe, column_name):
    """
    Function to return unique values along with their counts in a specified column of a DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column for which unique values with counts are desired.

    Returns:
    dict: A dictionary where keys are the unique values and values are their corresponding counts.
    """
    unique_counts = dataframe[column_name].value_counts().to_dict()
    return unique_counts


def check_series_values(series):
    """
    Function to check if all values in a pandas Series are 0 or 1.

    Parameters:
    series (pandas.Series): The pandas Series to check.

    Returns:
    bool: True if all values are 0 or 1, False otherwise.
    """
    # Check if all values are either 0 or 1
    return series.isin([0, 1]).all()


def custom_agg(column):
    """
    Custom aggregation function to return the last value in the column.

    Parameters:
        column (pandas.Series): A pandas Series representing a column in a DataFrame.

    Returns:
        object: The last value in the column if it's not empty, otherwise None.
    """
    return column.iloc[-1] if not column.empty else None


def transform_to_equal_interval(dataframe, frequency="10T"):
    """
    Function to transform a DataFrame to an equal interval format.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to be transformed.
        frequency (str): The frequency interval to resample the DataFrame to (default is '10T' for 10 minutes).

    Returns:
        pandas.DataFrame: DataFrame transformed to an equal interval format.
    """
    # Convert the 'Time' column to datetime format
    dataframe["Time"] = pd.to_datetime(dataframe["Time"])

    # Set the 'Time' column as the index of the DataFrame
    dataframe.set_index("Time", inplace=True)

    # Pivot the DataFrame to create a new DataFrame with columns as 'ItemName', rows as time intervals,
    # and values as the last value in each time interval using the custom_agg function
    transformed_df = dataframe.pivot_table(
        index=pd.Grouper(freq=frequency), columns="ItemName", values="Value", aggfunc=custom_agg, fill_value=None
    )

    # Replace NaN values with None for better compatibility
    transformed_df = transformed_df.where(pd.notnull(transformed_df), None)

    return transformed_df


def select_columns(dataframe, column_list):
    """
    This function selects only columns in the DataFrame that are present in both the DataFrame
    and the provided list of column names.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to filter columns from.
        column_list (list of str): List of column names to match against DataFrame columns.

    Returns:
        pandas.DataFrame: DataFrame with columns present in both the DataFrame and the provided
                          list of column names.
    """

    # Create a list of common columns that are present both in the DataFrame and the provided column list
    common_columns = [col for col in dataframe.columns if col in column_list]

    # Return a DataFrame containing only the common columns
    return dataframe[common_columns]


def remove_rows_starting_with_string(dataframe, column_name, word):
    """
    Function to remove rows from a DataFrame where a specific column starts with word.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame from which rows are to be removed.
    column_name (str): The name of the column to check for values starting with word.
    word (str): The word to check for at the beginning of the values in the specified column.

    Returns:
    pandas.DataFrame: DataFrame with rows removed where the specified column starts with word.
    """
    # Convert the values in the specified column to string if they are not already
    dataframe[column_name] = dataframe[column_name].astype(str)

    # Filter rows where the specified column starts with word
    filtered_dataframe = dataframe[~dataframe[column_name].str.startswith(word)]

    return filtered_dataframe


def extract_location_and_activity(row):
    """
    This function extracts location and activity information from a given row.

    Parameters:
        row (Series): A row from a DataFrame containing a 'label' column.

    Returns:
        Series: A Series containing the extracted 'location' and 'activity' information.
    """

    # Check if the 'label' column is not empty
    if row["label"]:
        # Split the 'label' string using ':' as separator and get the second part,
        # then split that part using '|' as separator to extract location and activity
        data = row["label"].split(":")[1].split("|")

        # Return a Series with 'location' and 'activity' as the index and extracted values as data
        return pd.Series(data, index=["location", "activity"])


def filter_rows_based_on_given_values_list(df, values_list):
    """
    This function filters rows in a DataFrame based on a given list of values.

    Parameters:
        df (DataFrame): The DataFrame to filter.
        values_list (list): The list of values to filter rows by.

    Returns:
        DataFrame: A new DataFrame containing only rows where the 'ItemName' column matches
                   one of the values in the provided list.
    """

    # Filter the DataFrame based on whether the 'ItemName' column matches any value in the given list
    filtered_df = df[df["ItemName"].isin(values_list)]

    # Return the filtered DataFrame
    return filtered_df


def get_motion_count_from_presence_dataframe(df):
    """
    This function calculates motion counts from a DataFrame containing presence data.

    Parameters:
        df (DataFrame): The DataFrame containing presence data.

    Returns:
        DataFrame: A new DataFrame with motion counts aggregated over time intervals.
    """

    # Convert the 'Time' column to datetime format
    df["Time"] = pd.to_datetime(df["Time"])

    # Define the time interval for aggregation
    interval = "1T"  # 1 minute intervals

    # Define the presence items to consider
    presence_items = ["kitchen_presence", "bathroom_presence", "office_presence", "bedroom_presence"]

    # Filter the DataFrame to include only rows related to presence items
    filtered_df = filter_rows_based_on_given_values_list(df, presence_items)

    # Create a copy of the filtered DataFrame to avoid modifying the original DataFrame
    filtered_df_copy = filtered_df.copy()

    # Replace 'ON' with 1 and 'OFF' with 0 for easier computation
    filtered_df_copy.replace({"ON": 1, "OFF": 0}, inplace=True)

    # Transform the DataFrame to have equal intervals based on the specified frequency
    a = transform_to_equal_interval(filtered_df_copy, frequency=interval)

    # Resample the DataFrame to the defined interval and fill missing values with the previous value
    df_resampled = a.resample(interval).asfreq()

    # Fill missing values with forward fill method and fill remaining NaNs with 0
    resampled_cleaned_df = df_resampled.ffill().fillna(0)

    # Resample the DataFrame to 10-minute intervals and calculate the sum of motion counts within each interval
    output_df = resampled_cleaned_df.resample("10T").sum()

    # Return the final DataFrame containing aggregated motion counts
    return output_df


def get_cleaned_sensor_dataframe(df):
    """
    Function to clean and prepare a sensor DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing sensor data.

    Returns:
        pandas.DataFrame: Cleaned and prepared DataFrame with sensor data.
    """
    # Remove rows starting with "STOP" in the 'Value' column
    df = remove_rows_starting_with_string(df.copy(), "Value", "STOP")

    # Replace "ON" with 1 and "OFF" with 0 in the DataFrame
    df.replace({"ON": 1, "OFF": 0}, inplace=True)

    # Transform DataFrame to equal interval format with a frequency of 10 minutes
    pivot_df = transform_to_equal_interval(df.copy(), frequency="10T")

    # Define rooms and sensors
    rooms = ["kitchen", "bedroom", "bathroom", "livingroom", "office"]
    sensors = ["CO2", "temperature", "luminosity", "humidity"]

    # Create a list of columns to select
    columns = ["label"]
    for room in rooms:
        for sensor in sensors:
            column = room + "_" + sensor
            columns.append(column)

    # Select minimal columns from the pivoted DataFrame
    minimal_df = select_columns(pivot_df.copy(), columns)

    # Extract location and activity from the 'label' column
    minimal_df[["location", "activity"]] = minimal_df.apply(extract_location_and_activity, axis=1)

    # Fill missing values using forward and backward filling
    minimal_df = minimal_df.fillna(method="ffill").fillna(method="bfill")

    return minimal_df


# Read CSV file into a DataFrame
df = read_csv_to_dataframe(base_path)

# Clean and prepare the sensor DataFrame
sensor_df = get_cleaned_sensor_dataframe(df)

# Calculate motion count from the presence DataFrame
motion_count_df = get_motion_count_from_presence_dataframe(df)

# Merge the sensor DataFrame and motion count DataFrame based on the 'Time' column
merged_df = pd.merge(sensor_df, motion_count_df, on="Time", how="inner")

# merged_df.to_csv("minimal_orange_motion_count_filled_10T.csv")

rooms = ["kitchen", "bedroom", "bathroom", "livingroom", "office"]

sensors = ["presence", "CO2", "temperature", "luminosity", "humidity"]


"""
Saving CSV for each sensor data
"""
import os

if not os.path.exists("sensors"):
    os.mkdir("sensors")


for sensor in sensors:
    columns = ["location", "activity"]
    for room in rooms:
        column = room + "_" + sensor
        columns.append(column)
    data = select_columns(merged_df.copy(), columns)
    data.to_csv(f"sensors/{sensor}.csv")


def presence_confidence(row):
    """
    Calculate the presence confidence based on the presence of sensors in different rooms.

    Parameters:
        row (pandas.Series): A row from a DataFrame containing sensor data.

    Returns:
        str or None: The room with the highest presence confidence or None if confidence is low.
    """
    # Get columns representing presence sensors in different rooms
    columns = [col for col in row.index if "_presence" in col]

    # Extract presence data from the row and convert it to a numpy array of floats
    data = row[columns].to_numpy()
    data = data.astype(float)

    # If all presence values are zero, return None indicating low confidence
    if np.all(data == 0):
        return None

    # If more than one sensor is active, check the difference in their presence values
    if np.count_nonzero(data) > 1:
        # Sort presence values in descending order
        arr_sorted = np.sort(data)[::-1]
        # Calculate the difference between the two highest presence values
        difference = arr_sorted[0] - arr_sorted[1]
        # If the difference is less than 3, return None indicating low confidence
        if difference < 3:
            return None

    # Get the index of the room with the highest presence value
    max_index = np.argmax(data)
    # Extract the room name from the column name
    room = columns[max_index].split("_")[0]
    return room


def clean_location(value):
    """
    Clean and format the location value.

    Parameters:
        value (str): The location value to clean.

    Returns:
        str: The cleaned and formatted location value.
    """
    # Remove underscores and convert to lowercase
    return value.replace("_", "").lower()


# Apply the presence_confidence function to each row of the DataFrame to predict the most confident room presence,
# and fill missing values forward to propagate the last valid observation.
merged_df["presence_prediction"] = merged_df.apply(presence_confidence, axis=1).ffill()

# Clean and format the 'location' column by applying the clean_location function to each value.
merged_df["location"] = merged_df["location"].apply(clean_location)

# Print the number of rows before filtering.
print("Rows Before Filtering :", merged_df.shape[0])

# Define the list of rooms where presence is expected.
"""
Q1 ) Why not we added livingroom in presence_rooms?
Ans. Cause in Orange4Home we have 2 presence sensors one for
    table and one for couch for livingroom. So to get correct
    metrices we are removing livingroom labels from location column.
"""
presence_rooms = ["kitchen", "bedroom", "bathroom", "office"]

# Filter the DataFrame to keep only rows where the 'location' column contains values from the presence_rooms list.
cleaned_merged_df = merged_df.loc[merged_df["location"].isin(presence_rooms)]


# Print the number of rows after filtering.
print("Rows After Filtering :", cleaned_merged_df.shape[0])

print(cleaned_merged_df[["location", "presence_prediction"]].head(10))


# Define the number of hidden states for the HMM
num_hidden_states = 6

# Initialize a Gaussian HMM model with specified parameters
model = hmm.GaussianHMM(n_components=num_hidden_states, covariance_type="full", n_iter=100)


# Drop columns that are not used as features for the model
X = cleaned_merged_df.drop(columns=["location", "presence_prediction", "activity", "label"])

# Convert the data in X to numeric format, coercing errors to NaN
X = X.apply(pd.to_numeric, errors="coerce")

# Fit the HMM model to the data
model.fit(X)

# Predict the hidden states for each observation in X
hidden_states = model.predict(X)

# Add the predicted hidden states as a new column in the DataFrame
cleaned_merged_df["hidden_state"] = hidden_states

# Initialize an empty dictionary to store the mapping
mapping = {}

# Iterate over each unique hidden state in the HMM Model
for state in cleaned_merged_df["hidden_state"].unique():
    # Filter the DataFrame to include only rows with the current hidden state
    subset = cleaned_merged_df[cleaned_merged_df["hidden_state"] == state]
    # Find the most common location associated with the current hidden state
    most_common_location = subset["location"].mode().iloc[0]
    # Add the mapping between the hidden state and its most common location to the dictionary
    mapping[state] = most_common_location

# Print the mapping dictionary
print("HMM State to Label Mapping :", mapping)

# Add a new column "hidden_label" to the DataFrame by mapping each hidden state to its most common location
cleaned_merged_df["hidden_label"] = cleaned_merged_df["hidden_state"].map(mapping)


def calculate_metrics(actual, predicted):
    """
    Calculate multiple machine learning metrics based on actual truth and predictions.

    Parameters:
        actual (array-like): Array of true labels.
        predicted (array-like): Array of predicted labels.

    Returns:
        dict: Dictionary containing multiple metrics.
    """
    # Initialize an empty dictionary to store the computed metrics
    metrics = {}

    # Calculate accuracy and store it in the metrics dictionary
    metrics["Accuracy"] = accuracy_score(actual, predicted)

    # for average parameter for metrices
    average = "macro"

    # Calculate precision and store it in the metrics dictionary
    metrics["Precision"] = precision_score(actual, predicted, average=average)

    # Calculate recall and store it in the metrics dictionary
    metrics["Recall"] = recall_score(actual, predicted, average=average)

    # Calculate F1-score and store it in the metrics dictionary
    metrics["F1 Score"] = f1_score(actual, predicted, average=average)

    # Defining the Comfusion Matrix Labels
    presence_rooms = ["kitchen", "bedroom", "bathroom", "office"]

    # Calculate confusion matrix and store it in the metrics dictionary
    metrics["Confusion Matrix"] = confusion_matrix(actual, predicted, labels=presence_rooms)

    # Return the computed metrics
    return metrics


# METRICES FOR RULE BASED SYSTEM

print("METRICES FOR RULE BASED SYSTEM")
rule_metrix = calculate_metrics(cleaned_merged_df["location"], cleaned_merged_df["presence_prediction"])
print(rule_metrix)

# METRICES FOR HMM Model

print("METRICES FOR HMM Model")
hmm_metrix = calculate_metrics(cleaned_merged_df["location"], cleaned_merged_df["hidden_label"])
print(hmm_metrix)
