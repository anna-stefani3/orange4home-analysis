import numpy as np
from statistics import mean


def get_indexes_of_motion(row):
    """
    Retrieve indexes where the maximum value occurs in a pandas Series.

    Parameters:
    row (pandas.Series): The pandas Series to analyze.

    Returns:
    list: List of indexes where the maximum value occurs.
    """
    return row.index[row == max(row)].tolist()


def get_location(row):
    """
    Determine the location based on motion detection columns.

    Parameters:
    row (pandas.Series): Row containing motion detection data.

    Returns:
    str: Detected location or "unknown".
    """
    data = row[["bedroom_presence", "kitchen_presence", "bathroom_presence", "livingroom_presence_table"]]
    indexes = get_indexes_of_motion(data)

    if max(data) == 0:
        return "unknown"

    if len(indexes) == 1:
        return indexes[0].split("_")[0]
    else:
        return np.NaN


def get_max(values):
    """
    Retrieve the maximum value from a list, excluding the last element.

    Parameters:
    values (list): List of numeric values.

    Returns:
    float: Maximum value excluding the last element.
    """
    if values[:-1]:
        return max(values[:-1])
    else:
        return values[0]


def get_mean(values):
    """
    Calculate the mean of a list, excluding the last element.

    Parameters:
    values (list): List of numeric values.

    Returns:
    float: Mean of the values excluding the last element.
    """
    if values[:-1]:
        return mean(values[:-1])
    else:
        return values[0]


def is_cooking(humidity_history, temperature_history, noise):
    """
    Determine if the activity is cooking based on historical data.

    Parameters:
    humidity_history (list): List of humidity values.
    temperature_history (list): List of temperature values.
    noise (int): Noise level.

    Returns:
    str: "cooking" if cooking activity is detected, "unknown" otherwise.
    """
    current_humidity = humidity_history[-1]
    is_humidity_increasing = current_humidity > get_max(humidity_history)
    humidity_threshold = 40
    is_high_humidity = current_humidity > humidity_threshold

    current_temperature = temperature_history[-1]
    is_high_temperature = current_temperature > get_max(temperature_history)

    if is_high_humidity or is_humidity_increasing or noise >= 2 or is_high_temperature:
        return "cooking"
    else:
        return "unknown"


def is_sleeping(co2_history, luminosity, motion, noise):
    """
    Determine if the activity is sleeping based on luminosity, motion, and noise.

    Parameters:
    co2_history (list[float]): CO2 level.
    luminosity (int): Luminosity level.
    motion (int): Motion level.
    noise (int): Noise level.

    Returns:
    str: "sleeping" if sleeping activity is detected, "unknown" otherwise.
    """
    current_co2 = co2_history[-1]
    is_co2_increasing = current_co2 > get_max(co2_history)

    luminosity_sleep_threshold = 50
    is_low_luminosity = luminosity <= luminosity_sleep_threshold

    if is_low_luminosity and (noise <= 2 or motion <= 2 or is_co2_increasing):
        return "sleeping"
    else:
        return "unknown"


def is_bathing(humidity_history, motion):
    """
    Determine if the activity is bathing or toileting based on historical humidity and motion.

    Parameters:
    humidity_history (list): List of humidity values.
    motion (int): Motion level.

    Returns:
    str: "bathing" if bathing activity is detected, "toileting" otherwise.
    """
    current_humidity = humidity_history[-1]
    is_humidity_increasing = current_humidity > get_max(humidity_history)
    humidity_threshold = 40
    is_high_humidity = current_humidity > humidity_threshold
    if motion > 10 or is_high_humidity or is_humidity_increasing:
        return "bathing"
    else:
        return "toileting"


def is_eating(motion, luminosity):
    """
    Determine if the activity is eating based on motion and luminosity.

    Parameters:
    motion (int): Motion level.
    luminosity (int): Luminosity level.

    Returns:
    str: "eating" if eating activity is detected, "unknown" otherwise.
    """
    if luminosity > 400 and motion > 8:
        return "eating"
    else:
        return "unknown"


def get_activity_label(df, index):
    """
    Determine the activity label based on location prediction and sensor data.

    Parameters:
    df (DataFrame): DataFrame containing sensor data and predictions.
    index (int): Index of the current data point.

    Returns:
    str: Activity label (e.g., "sleeping", "bathing", "cooking", "eating", "unknown").
    """
    location = df["location_prediction"][index]

    if location == "bedroom":
        co2_history = df["bedroom_CO2"][max(index - 10, 0) : index + 1].to_list()
        luminosity = df["bedroom_luminosity"][index]
        motion = df["bedroom_presence"][index]
        noise = df["bedroom_noise"][index]
        return is_sleeping(co2_history, luminosity, motion, noise)

    elif location == "bathroom":
        humidity_history = df["bathroom_humidity"][max(index - 10, 0) : index + 1].to_list()
        motion = df["bathroom_presence"][index]
        return is_bathing(humidity_history, motion)

    elif location == "kitchen":
        humidity_history = df["kitchen_humidity"][max(index - 10, 0) : index + 1].to_list()
        temperature_history = df["kitchen_temperature"][max(index - 10, 0) : index + 1].to_list()
        noise = df["kitchen_noise"][index]
        return is_cooking(humidity_history, temperature_history, noise)

    elif location == "livingroom":
        luminosity = df["livingroom_luminosity"][index]
        motion = df["livingroom_presence_table"][index]
        return is_eating(motion, luminosity)

    else:
        return "unknown"
