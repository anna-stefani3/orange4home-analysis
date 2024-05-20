# Define file paths
BASE_PATH = "o4h_all_events.csv"
SAVE_PATH = "o4h_activity_dataframe.csv"


# lists for activities and locations
ACTIVITIES_LIST = ["sleeping", "cooking", "bathing", "toileting", "eating", "unknown"]
LOCATIONS_LIST = ["bedroom", "bathroom", "kitchen", "livingroom", "unknown"]


# Define a dictionary for mapping activities
ACTIVITY_MAPPING = {
    "Using_the_toilet": "toileting",
    "Napping": "sleeping",
    "Showering": "bathing",
    "Using_the_sink": "toileting",
    "Preparing": "cooking",
    "Eating": "eating",
    "Cooking": "cooking",
    "Washing_the_dishes": "cooking",
}

# Define a dictionary for mapping the location names
LOCATION_MAPPING = {"Bathroom": "bathroom", "Living_room": "livingroom", "Kitchen": "kitchen", "Bedroom": "bedroom"}

# Define a dictionary for mapping the location to int (Needed For Decision Tree)
LOCATION_INT_MAPPING = {"bathroom": 1, "livingroom": 2, "kitchen": 3, "bedroom": 4, "unknown": 5}

# Mapping activities to locations
ACTIVITY_LOCATION_MAPPING = {
    "sleeping": "bedroom",
    "cooking": "kitchen",
    "bathing": "bathroom",
    "toileting": "bathroom",
    "eating": "livingroom",
}
