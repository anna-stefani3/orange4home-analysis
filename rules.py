import numpy as np
from statistics import mean


def get_indexes_of_motion(row):
    """
    Function to return the indexes where the value is 1 in a pandas Series.

    Parameters:
    series (pandas.Series): The pandas Series to search.

    Returns:
    list: A list of indexes where the value is greater than 0.
    """
    # Use the index attribute of the Series along with boolean indexing to get the indexes where the value is 1
    return row.index[row == max(row)].tolist()


import numpy as np


def get_location(row):
    # Extract relevant data columns from the row
    data = row[["bedroom_presence", "kitchen_presence", "bathroom_presence", "livingroom_presence_table"]]

    # Get indexes of columns with motion detected
    indexes = get_indexes_of_motion(data)

    # Check if all values are zero
    if max(data) == 0:
        return "unknown"

    # Check if only one motion is detected
    if len(indexes) == 1:
        # Extract the location from the index
        return indexes[0].split("_")[0]
    else:
        # Return NaN if multiple motions are detected
        return np.NaN


def get_max(values):
    """
    Get the maximum value from a list of values, excluding the last value.

    Parameters:
    values (list): List of numeric values

    Returns:
    float: Maximum value from the list, excluding the last value
    """
    # Check if the list is not empty
    if values[:-1]:
        # Return the maximum value excluding the last value
        return max(values[:-1])
    else:
        # Return the first value if the list is empty
        return values[0]


from statistics import mean


def get_mean(values):
    """
    Calculate the mean of a list of values, excluding the last value.

    Parameters:
    values (list): List of numeric values

    Returns:
    float: Mean of the values, excluding the last value
    """
    # Check if the list is not empty
    if values[:-1]:
        # Return the mean of the values excluding the last value
        return mean(values[:-1])
    else:
        # Return the first value if the list is empty
        return values[0]


def is_cooking_or_eating(humidity_history, temperature_history, sound_history):
    """
    Determine if the current activity is cooking or eating based on historical data.

    Parameters:
    humidity_history (list): List of humidity values
    temperature_history (list): List of temperature values
    sound_history (list): List of sound values

    Returns:
    str: "cooking" if cooking activity is detected, "eating" otherwise
    """
    # Get the current humidity value
    current_humidity = humidity_history[-1]

    # Check if the current humidity is higher than the maximum humidity
    is_high_humidity = current_humidity > get_max(humidity_history)

    # Get the current temperature value
    current_temperature = temperature_history[-1]

    # Check if the current temperature is increasing
    is_temperature_increasing = current_temperature > get_mean(temperature_history)

    # Get the current sound value
    current_sound = sound_history[-1]

    # Check if the current sound level is higher than the maximum sound level
    is_high_noise = current_sound > get_max(sound_history)

    # Determine the activity based on the conditions
    if is_high_humidity or is_temperature_increasing or is_high_noise:
        return "cooking"
    else:
        return "eating"


def is_sleeping(co2_history, motion):
    """
    Determine if the current activity is sleeping based on CO2 levels and motion.

    Parameters:
    co2_history (list): List of CO2 values
    motion (int): Motion detection value (0 for no motion, 1 for motion)

    Returns:
    str: "sleeping" if sleeping activity is detected, "unknown" otherwise
    """
    # Get the current CO2 value
    current_co2 = co2_history[-1]

    # Check if the current CO2 level is higher than the maximum CO2 level
    is_high_co2 = current_co2 > get_max(co2_history)

    # Determine the activity based on the conditions
    if is_high_co2 and motion == 0:
        return "sleeping"
    else:
        return "unknown"


def is_bathing_or_toileting(humidity_history, co2_history, motion_history):
    """
    Determine if the current activity is bathing or toileting based on historical data.

    Parameters:
    humidity_history (list): List of humidity values
    co2_history (list): List of CO2 values
    motion_history (list): List of motion count values

    Returns:
    str: "bathing" if bathing activity is detected, "toileting" otherwise
    """
    # Get the current humidity value
    current_humidity = humidity_history[-1]

    # Check if the current humidity is higher than the maximum humidity
    is_high_humidity = current_humidity > get_max(humidity_history)

    # Get the current CO2 value
    current_co2 = co2_history[-1]

    # Check if the current CO2 level is higher than the maximum CO2 level
    is_high_co2 = current_co2 > get_max(co2_history)

    # Get the current motion count value
    current_motion_count = motion_history[-1]

    # Check if the current motion count is higher than the maximum motion count
    is_high_motion_count = current_motion_count > get_max(motion_history)

    # Determine the activity based on the conditions
    if (is_high_humidity or is_high_co2) and is_high_motion_count:
        return "bathing"
    else:
        return "toileting"


def is_socialising_or_eating(co2_history, motion_history):
    """
    Determine if the current activity is socialising or eating based on CO2 levels and motion count.

    Parameters:
    co2_history (list): List of CO2 values
    motion_history (list): List of motion count values

    Returns:
    str: "socialising" if socialising activity is detected, "eating" otherwise
    """
    # Get the current CO2 value
    current_co2 = co2_history[-1]

    # Check if the current CO2 level is higher than the maximum CO2 level
    is_high_co2 = current_co2 > get_max(co2_history)

    # Get the current motion count value
    current_motion_count = motion_history[-1]

    # Check if the current motion count is higher than the maximum motion count
    is_high_motion_count = current_motion_count > get_max(motion_history)

    # Determine the activity based on the conditions
    if is_high_co2 and is_high_motion_count:
        return "socialising"
    else:
        return "eating"


def get_activity_label(df, index):
    """
    Determine the activity label based on the predicted location and historical sensor data.

    Parameters:
    df (DataFrame): DataFrame containing sensor data and predictions
    index (int): Index of the current data point

    Returns:
    str: Activity label (e.g., "sleeping", "bathing", "cooking", "socialising", "unknown")
    """
    # Retrieve the predicted location at the given index
    location = df["location_prediction"][index]

    # Determine the activity based on the predicted location
    if location == "bedroom":
        co2_history = df["bedroom_CO2"][max(index - 10, 0) : index + 1].to_list()
        motion = df["bedroom_presence"][index]
        return is_sleeping(co2_history, motion)

    elif location == "bathroom":
        co2_history = df["bathroom_CO2"][max(index - 10, 0) : index + 1].to_list()
        humidity_history = df["bathroom_humidity"][max(index - 10, 0) : index + 1].to_list()
        motion_history = df["bathroom_presence"][max(index - 10, 0) : index + 1].to_list()
        return is_bathing_or_toileting(humidity_history, co2_history, motion_history)

    elif location == "kitchen":
        humidity_history = df["kitchen_humidity"][max(index - 10, 0) : index + 1].to_list()
        temperature_history = df["kitchen_temperature"][max(index - 10, 0) : index + 1].to_list()
        sound_history = df["kitchen_noise"][max(index - 10, 0) : index + 1].to_list()
        return is_cooking_or_eating(humidity_history, temperature_history, sound_history)

    elif location == "livingroom":
        co2_history = df["livingroom_CO2"][max(index - 10, 0) : index + 1].to_list()
        motion_history = df["livingroom_presence_table"][max(index - 10, 0) : index + 1].to_list()
        return is_socialising_or_eating(co2_history, motion_history)

    else:
        return "unknown"
