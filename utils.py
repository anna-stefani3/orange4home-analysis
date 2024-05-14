import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier


# Function to read a CSV file into a DataFrame
def read_csv_to_dataframe(csv_file):
    """
    Read a CSV file into a DataFrame.

    Parameters:
    csv_file (str): File path or name of the CSV file.

    Returns:
    pandas.DataFrame: DataFrame containing the CSV data.
    """
    dataframe = pd.read_csv(csv_file)
    return dataframe


# Function to extract activity duration data
def get_activity_duration_data(df):
    """
    Extract activity duration data from the DataFrame.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with activity duration data.
    """
    # Filtering activity rows
    activities = df[df["ItemName"] == "label"]

    # Splitting value into different columns
    activities[["case", "location", "activity"]] = activities["Value"].str.split(":|\\|", expand=True)

    # Selecting required columns
    activities = activities[["Time", "ItemName", "case", "location", "activity"]]

    # Convert time to datetime format
    activities["Time"] = pd.to_datetime(activities["Time"])

    # Filtering start and stop events
    start_events = activities[activities["case"] == "START"].reset_index(drop=True)
    stop_events = activities[activities["case"] == "STOP"].reset_index(drop=True)

    # Calculating duration in seconds
    result = (stop_events["Time"] - start_events["Time"]).dt.total_seconds()

    # Creating new DataFrame for activity duration
    activity_duration_df = pd.DataFrame(
        {
            "Start Time": start_events["Time"],
            "Stop Time": stop_events["Time"],
            "Activity": start_events["activity"],
            "Time Duration(s)": result,
        }
    )

    # Saving to CSV named "activity_duration_monitoring_data.csv"
    activity_duration_df.to_csv("activity_duration_monitoring_data.csv")

    return activity_duration_df


# Function to get unique values from a column
def unique_values_in_column(dataframe, column_name):
    """
    Get unique values from a DataFrame column.

    Parameters:
    dataframe (pandas.DataFrame): Input DataFrame.
    column_name (str): Name of the column.

    Returns:
    list: List of unique values.
    """
    unique_values = dataframe[column_name].unique().tolist()
    return unique_values


# Function to get unique values with counts
def unique_values_with_count(dataframe, column_name):
    """
    Get unique values with their counts from a DataFrame column.

    Parameters:
    dataframe (pandas.DataFrame): Input DataFrame.
    column_name (str): Name of the column.

    Returns:
    dict: Dictionary with unique values as keys and counts as values.
    """
    unique_counts = dataframe[column_name].value_counts().to_dict()
    return unique_counts


# Function to check if Series values are 0 or 1
def check_series_values(series):
    """
    Check if all values in a Series are either 0 or 1.

    Parameters:
    series (pandas.Series): Input Series.

    Returns:
    bool: True if all values are 0 or 1, False otherwise.
    """
    return series.isin([0, 1]).all()


# Custom aggregation function to get the last value
def custom_agg(column):
    """
    Custom aggregation function to get the last value in a column.

    Parameters:
    column (pandas.Series): Input Series.

    Returns:
    object: Last value in the Series.
    """
    return column.iloc[-1] if not column.empty else None


# Function to transform DataFrame to equal interval format
def transform_to_equal_interval(dataframe, frequency="1T"):
    """
    Transform DataFrame to an equal interval format.

    Parameters:
    dataframe (pandas.DataFrame): Input DataFrame.
    frequency (str): Frequency interval (default is '1T' for 1 minute).

    Returns:
    pandas.DataFrame: Transformed DataFrame.
    """
    # Convert time to datetime format
    dataframe["Time"] = pd.to_datetime(dataframe["Time"])

    # Set time as index
    dataframe.set_index("Time", inplace=True)

    # Pivot DataFrame for equal intervals
    transformed_df = dataframe.pivot_table(
        index=pd.Grouper(freq=frequency), columns="ItemName", values="Value", aggfunc=custom_agg, fill_value=None
    )

    # Replace NaN values
    transformed_df = transformed_df.where(pd.notnull(transformed_df), None)

    return transformed_df


# Function to select specified columns
def select_columns(dataframe, column_list):
    """
    Select specified columns from a DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): Input DataFrame.
    column_list (list of str): List of column names to select.

    Returns:
    pandas.DataFrame: DataFrame with selected columns.
    """
    common_columns = [col for col in dataframe.columns if col in column_list]
    return dataframe[common_columns]


# Function to remove rows starting with a specific string
def remove_rows_starting_with_string(dataframe, column_name, word):
    """
    Remove rows from a DataFrame where a specific column starts with a given word.

    Parameters:
    dataframe (pandas.DataFrame): Input DataFrame.
    column_name (str): Name of the column to check.
    word (str): The word to check for.

    Returns:
    pandas.DataFrame: DataFrame with specified rows removed.
    """
    dataframe[column_name] = dataframe[column_name].astype(str)
    filtered_dataframe = dataframe[~dataframe[column_name].str.startswith(word)]
    return filtered_dataframe


# Function to extract location and activity
def extract_location_and_activity(row):
    """
    Extract location and activity information from a given row.

    Parameters:
    row (pandas.Series): Input row.

    Returns:
    pandas.Series: Series containing extracted 'location' and 'activity' information.
    """
    if row["label"]:
        data = row["label"].split(":")[1].split("|")
        return pd.Series(data, index=["location", "activity"])


# Function to filter rows based on given values list
def filter_rows_based_on_given_values_list(df, values_list, column="ItemName"):
    """
    Filter rows in a DataFrame based on a given list of values.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    values_list (list): List of values to filter by.
    column (str): Name of the column to filter.

    Returns:
    pandas.DataFrame: DataFrame containing filtered rows.
    """
    filtered_df = df[df[column].isin(values_list)]
    return filtered_df


# Function to calculate motion counts
def get_motion_count_from_presence_dataframe(df, frequency="1T"):
    """
    Calculate motion counts from a DataFrame containing presence data.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with aggregated motion counts.
    """
    df["Time"] = pd.to_datetime(df["Time"])
    interval = "5S" if frequency == "1T" else "1T"
    presence_items = ["kitchen_presence", "bathroom_presence", "livingroom_presence_table", "bedroom_presence"]
    filtered_df = filter_rows_based_on_given_values_list(df, presence_items)
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy.replace({"ON": 1, "OFF": 0}, inplace=True)
    a = transform_to_equal_interval(filtered_df_copy, frequency=interval)
    df_resampled = a.resample(interval).asfreq()
    resampled_cleaned_df = df_resampled.ffill().fillna(0)
    output_df = resampled_cleaned_df.resample(frequency).sum()
    return output_df


# Function to clean and prepare sensor data
def get_cleaned_sensor_dataframe(df, frequency="1T"):
    """
    Clean and prepare sensor data.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    df = remove_rows_starting_with_string(df.copy(), "Value", "STOP")
    df.replace({"ON": 1, "OFF": 0}, inplace=True)
    pivot_df = transform_to_equal_interval(df.copy(), frequency=frequency)
    rooms = ["kitchen", "bedroom", "bathroom", "livingroom"]
    sensors = ["CO2", "temperature", "luminosity", "humidity", "noise"]
    columns = ["label", "livingroom_table_noise"]
    for room in rooms:
        for sensor in sensors:
            column = room + "_" + sensor
            columns.append(column)
    minimal_df = select_columns(pivot_df.copy(), columns)
    minimal_df[["location", "activity"]] = minimal_df.apply(extract_location_and_activity, axis=1)
    minimal_df = minimal_df.fillna(method="ffill").fillna(method="bfill")
    return minimal_df


# Function to split DataFrame into features and labels
def get_feature_and_label(df, labels=["location", "location_int", "activity"]):
    """
    Split DataFrame into features and labels.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame, pandas.DataFrame: Features and labels.
    """
    X = df.drop(columns=labels)
    y = df[labels]
    return X, y


# Function to balance data using SMOTE
def get_balanced_data(X, y):
    """
    Balance data using SMOTE.

    Parameters:
    X (pandas.DataFrame): Features.
    y (pandas.DataFrame): Labels.

    Returns:
    pandas.DataFrame, pandas.DataFrame: Balanced features and labels.
    """
    try:
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y["activity"])
    except:
        smote = SMOTE(k_neighbors=1)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.DataFrame(y_resampled, columns=["location", "activity"])
    return X_resampled_df, y_resampled_df


# Function to calculate evaluation metrics
def calculate_metrics(actual, predicted, labels=None):
    """
    Calculate machine learning metrics based on actual truth and predictions.

    Parameters:
    actual (array-like): True labels.
    predicted (array-like): Predicted labels.
    labels (list): List of labels.

    Returns:
    dict: Dictionary containing metrics.
    """
    metrics = {}
    # metrics["Accuracy"] = accuracy_score(actual, predicted)
    # average = "macro"
    # metrics["Precision"] = precision_score(actual, predicted, average=average)
    # metrics["Recall"] = recall_score(actual, predicted, average=average)
    # metrics["F1 Score"] = f1_score(actual, predicted, average=average)
    # metrics["Confusion Matrix"] = confusion_matrix(actual, predicted, labels=labels)
    metrics["Classification Report"] = classification_report(actual, predicted, labels=labels)
    # metrics["labels"] = labels
    return metrics


# Function to train and test and then get the evaluation_metrics
def model_train_test_score(model, X_train, X_test, y_train, y_test, label_sequence=None):
    """
    Train a model on training data and calculate evaluation metrics on test data.

    Parameters:
    model (object): The machine learning model to train and evaluate.
    X_train (array-like): Features of the training data.
    X_test (array-like): Features of the test data.
    y_train (array-like): Labels of the training data.
    y_test (array-like): Labels of the test data.
    label_sequence (list, optional): Sequence of labels for confusion matrix (default is None).

    Returns:
    object, dict: Trained model and evaluation metrics.
    """
    # Fit the model on training data
    model.fit(X_train, y_train)

    # Predict labels for test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    evaluation_metrics = calculate_metrics(y_test, y_pred, labels=label_sequence)

    return model, evaluation_metrics, y_pred


# Function to train a model and calculate evaluation metrics
def model_train_test_split(model, X_resampled, y_resampled, labels=None):
    """
    Train a model on resampled data using train-test split.

    Parameters:
    model (object): Model to train.
    X_resampled (array-like): Resampled features.
    y_resampled (array-like): Resampled labels.
    labels (list): List of labels.

    Returns:
    object, dict: Trained model and evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled["activity"], test_size=0.66, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluation_metrics = calculate_metrics(y_test, y_pred, labels=labels)
    return model, evaluation_metrics


# Function to get decision tree structure
def get_decision_tree_structure(model, feature_names):
    """
    Get decision tree structure as text.

    Parameters:
    model (object): Trained decision tree model.
    feature_names (list): List of feature names.

    Returns:
    str: Decision tree structure as text.
    """
    tree_structure = export_text(model, feature_names=feature_names)
    return tree_structure


# Function to print evaluation metrics
def print_metrices(metrics):
    """
    Print evaluation metrics.

    Parameters:
    metrics (dict): Evaluation metrics.

    Returns:
    None
    """
    for metric in metrics:
        if metric == "labels":
            continue
        if "Confusion" in metric:
            print(f"{metric} | Sequence -> {metrics.get('labels')}\n")
            print(metrics[metric])
        elif "Classification" in metric:
            print(f"{metric}\n")
            print(metrics[metric])
        else:
            print(f"{metric: <12} - {metrics[metric] * 100:.2f} %")


def get_day_wise_group_df(grouped_daywise_data, group_ids, start_day, end_day):
    """
    Concatenates DataFrames corresponding to specific group IDs from a grouped DataFrame.

    Parameters:
    grouped_daywise_data (pandas GroupBy object): Grouped DataFrame by day-wise data.
    group_ids (list): List of group IDs to concatenate.
    start (int): Starting index of group IDs list.
    end (int): Ending index of group IDs list.

    Returns:
    pandas DataFrame: Merged DataFrame containing data for the specified group IDs.
    """

    # Initialize an empty list to store DataFrames for each group
    dfs = []

    # Iterate over each group ID in the specified range
    for group_name in group_ids[start_day:end_day]:
        # Get the DataFrame for the current group ID
        group_df = grouped_daywise_data.get_group(group_name)
        # Append the DataFrame to the list
        dfs.append(group_df)

    # Concatenate the DataFrames in the list
    grouped_df = pd.concat(dfs)

    # Reset index and return the merged grouped DataFrame
    return grouped_df.reset_index(drop=True)


def get_top_features_using_random_forest(features, labels, threshold=0.05):
    # Create a random forest classifier
    clf = RandomForestClassifier()

    # Train the classifier
    clf.fit(features, labels)

    # Get feature importances
    importances = clf.feature_importances_

    # Select features based on the threshold
    selected_features = features.columns[importances >= threshold]
    print("\n\n")
    print(f"Features Count: {len(features.columns)} | Selected Features: {len(selected_features)}")

    return selected_features


def get_top_features_using_RFE(features, labels, features_count=10):
    # Create a random forest classifier
    clf = RandomForestClassifier()

    # Features Selector
    selector = RFE(clf, n_features_to_select=features_count, step=10)

    # Train the selector
    selector.fit(features, labels)

    # Select features based on the threshold
    selected_features = features.columns[selector.support_]
    print("\n\n")
    print(f"Features Count: {len(features.columns)} | Selected Features: {len(selected_features)}")

    return selected_features


def get_top_features_using_RFECV(features, labels, min_features_count=5):
    # Create a random forest classifier
    clf = RandomForestClassifier()

    # Features Selector
    selector = RFECV(clf, min_features_to_select=min_features_count, step=10, n_jobs=-1)

    # Train the selector
    selector.fit(features, labels)

    # Select features based on the threshold
    selected_features = features.columns[selector.support_]
    print("\n\n")
    print(f"Features Count: {len(features.columns)} | Selected Features: {len(selected_features)}")

    return selected_features
