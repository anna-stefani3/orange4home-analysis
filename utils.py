import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


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


def get_activity_duration_data(df):
    # Filtering the activity rows
    activities = df[df["ItemName"] == "label"]

    # Splitting Value -> "STOP:Enterance|Entering" into ["STOP", "Entrance", "Entering"]
    activities[["case", "location", "activity"]] = activities["Value"].str.split(":|\\|", expand=True)

    # Selecting Required Columns
    activities = activities[["Time", "ItemName", "case", "location", "activity"]]

    # Convert "Time" column to datetime if it's not already in datetime format
    activities["Time"] = pd.to_datetime(activities["Time"])

    # Filter only the start events
    start_events = activities[activities["case"] == "START"].reset_index(drop=True)

    # Filter only the stop events
    stop_events = activities[activities["case"] == "STOP"].reset_index(drop=True)

    # Calculate time difference between start and stop events for each activity
    result = (stop_events["Time"] - start_events["Time"]).dt.total_seconds()

    # Create a DataFrame to store the results
    activity_duration_df = pd.DataFrame(
        {
            "Start Time": start_events["Time"],
            "Stop Time": stop_events["Time"],
            "Activity": start_events["activity"],
            "Time Duration(s)": result,
        }
    )

    print("Activities Duration Sample Data")
    print(activity_duration_df.head())

    # Display the resulting DataFrame
    activity_duration_df.to_csv("activity_duration_monitoring_data.csv")

    print("\n\nSAVED ACTIVITY DURATION DATA INTO 'activity_duration_monitoring_data.csv' FILE")
    return activity_duration_df


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


def transform_to_equal_interval(dataframe, frequency="1T"):
    """
    Function to transform a DataFrame to an equal interval format.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to be transformed.
        frequency (str): The frequency interval to resample the DataFrame to (default is '1T' for 1 minute).

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


def filter_rows_based_on_given_values_list(df, values_list, column="ItemName"):
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
    filtered_df = df[df[column].isin(values_list)]

    # Return the filtered DataFrame
    return filtered_df


def get_motion_count_from_presence_dataframe(df, frequency="1T"):
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
    interval = "5S" if frequency == "1T" else "1T"  # 5 Seconds intervals

    # Define the presence items to consider
    presence_items = ["kitchen_presence", "bathroom_presence", "livingroom_presence_table", "bedroom_presence"]

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

    # Resample the DataFrame to 1-minute intervals and calculate the sum of motion counts within each interval
    output_df = resampled_cleaned_df.resample(frequency).sum()

    # Return the final DataFrame containing aggregated motion counts
    return output_df


def get_cleaned_sensor_dataframe(df, frequency="1T"):
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

    # Transform DataFrame to equal interval format with a frequency of 1 minutes as default value
    pivot_df = transform_to_equal_interval(df.copy(), frequency=frequency)

    # Define rooms and sensors
    rooms = ["kitchen", "bedroom", "bathroom", "livingroom"]
    sensors = ["CO2", "temperature", "luminosity", "humidity", "noise"]

    # Create a list of columns to select
    columns = ["label", "livingroom_table_noise"]
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


def get_feature_and_label(merged_df):
    # features are in X and labels are in y
    X = merged_df.drop(columns=["location", "activity", "label"])
    y = merged_df["activity"]
    return X, y


def get_balanced_data(X, y):
    try:
        # Defining SMOTE
        smote = SMOTE()

        # Resample the dataset
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except:
        # Defining SMOTE when sample size is less than 6
        smote = SMOTE(k_neighbors=1)

        # Resample the dataset
        X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convert the resampled data back to a DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.DataFrame(y_resampled, columns=["activity"])

    return X_resampled_df, y_resampled_df


def model_train_test_split(model, X_resampled, y_resampled, labels=None):
    """
    Train a model on resampled data using train-test split and calculate evaluation metrics.

    Parameters:
    model (object): The model to train (e.g., DecisionTreeClassifier).
    X_resampled (array-like): The resampled features.
    y_resampled (array-like): The resampled labels.

    Returns:
    dict: Dictionary containing evaluation metrics (accuracy, precision, recall, F1-score).
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.8, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    CM = confusion_matrix(y_test, y_pred, labels=labels if labels else None)

    # Return evaluation metrics as a dictionary
    evaluation_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Confusion-Matrix": CM,
        "labels": labels,
    }

    return model, evaluation_metrics


from sklearn.tree import export_text


def get_decision_tree_structure(model, feature_names):
    """
    Get the decision tree structure as text.

    Parameters:
    model (object): The trained decision tree model.

    Returns:
    str: Decision tree structure as text.
    """
    # Get the decision tree structure as text
    tree_structure = export_text(model, feature_names=feature_names)

    return tree_structure


def calculate_metrics(actual, predicted, labels=None):
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

    # Calculate confusion matrix and store it in the metrics dictionary
    metrics["Confusion Matrix"] = confusion_matrix(actual, predicted, labels=labels)

    metrics["labels"] = labels

    # Return the computed metrics
    return metrics


def print_metrices(metrics):
    """
    Print the evaluation metrics.

    Parameters:
    metrics (dict): Dictionary containing evaluation metrics

    Returns:
    None
    """
    for metric in metrics:
        # Skip the "labels" key
        if metric == "labels":
            continue

        # Print confusion matrix separately
        if "Confusion" in metric:
            print(f"{metric} | Sequence -> {metrics.get('labels')}\n")
            print(metrics[metric])
        else:
            # Print other metrics with percentage format
            print(f"{metric: <12} - {metrics[metric] * 100:.2f} %")
