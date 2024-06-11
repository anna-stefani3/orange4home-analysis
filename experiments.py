from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from common_variables import ACTIVITIES_LIST, ACTIVITY_LOCATION_MAPPING, LOCATIONS_LIST, LOCATION_INT_MAPPING
from utils import model_train_test_score, print_metrices, calculate_metrics, validate_experiment
from rules import get_location, get_activity_label
from preprocessing import get_balanced_training_dataset

from sklearn import svm
from sklearn_crfsuite import CRF

from hmmlearn import hmm

# fmt: off
# Define parameters for the SVM
SVM_PARAMS = {
    "C": 1.0,                # Regularization parameter
    "kernel": "rbf",         # RBF kernel (Gaussian Kernel)
    "gamma": "scale"         # Kernel coefficient for "rbf", "poly" and "sigmoid"
}


CRF_PARAMS = {
    "algorithm":"lbfgs",
    "c1":0.1,
    "c2":0.1,
    "max_iterations":100,
    "all_possible_transitions":False
}

# Define parameters for the DecisionTreeClassifier
DT_PARAMS = {
    "criterion": "gini",        # Splitting criterion: "gini" or "entropy"
    "max_depth": 5,             # Maximum depth of the decision tree
    "min_samples_split": 2,     # Minimum samples required to split an internal node
    "min_samples_leaf": 1,      # Minimum samples required to be at a leaf node
    "max_features": "sqrt",     # Number of features to consider for the best split
    "random_state": 42,         # Random state for reproducibility
}
# fmt: on


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


def get_top_features_using_RFECV(features, labels, min_features_count=5):
    # Create a random forest classifier
    clf = RandomForestClassifier(random_state=42)

    # Features Selector
    selector = RFECV(clf, min_features_to_select=min_features_count, step=10, n_jobs=-1)

    # Train the selector
    selector.fit(features, labels)

    # Select features based on the threshold
    selected_features = features.columns[selector.support_]
    print("\n\n")
    print(f"Features Count: {len(features.columns)} | Selected Features: {len(selected_features)}")

    return selected_features


@validate_experiment
def apply_decision_tree_on_balanced_data(X_train, X_test, y_train, y_test):
    X_train, y_train = get_balanced_training_dataset(X_train, y_train)

    # Initialize the DecisionTreeClassifier with the specified parameters
    DT = DecisionTreeClassifier(**DT_PARAMS)

    # Get top features based on importance for activity prediction
    top_features = get_top_features_using_RFECV(X_train, y_train["activity"])
    X_train_activity_subset = X_train[top_features]
    X_test_activity_subset = X_test[top_features]

    # Train the Decision Tree classifier using the balanced dataset
    _, evaluation_results, _ = model_train_test_score(
        DT, X_train_activity_subset, X_test_activity_subset, y_train["activity"], y_test["activity"], ACTIVITIES_LIST
    )

    # Display the evaluation results for the trained Decision Tree model
    print("\n\n")
    print("Balanced DT - Activity Classification Results")
    print_metrices(evaluation_results)


def classify_activity_directly(model, X_train, X_test, y_train, y_test, mode="statistical"):
    """
    ##########################################
    Classify Activity Directly
    ##########################################
    """
    print("\n\nClassify Activity Directly")

    # Get top features based on importance for activity prediction
    top_features = get_top_features_using_RFECV(X_train, y_train["activity"])
    X_train_activity_subset = X_train[top_features]
    X_test_activity_subset = X_test[top_features]

    # Train and evaluate the model for activity prediction
    _, evaluation_results, _ = model_train_test_score(
        model,
        X_train_activity_subset,
        X_test_activity_subset,
        y_train["activity"],
        y_test["activity"],
        ACTIVITIES_LIST,
        mode=mode,
    )

    print("Activity Classification Results")
    print_metrices(evaluation_results)


@validate_experiment
def apply_decision_tree_to_classify_activity_directly(X_train, X_test, y_train, y_test):
    print("\n\nDECISION TREE - Classify Each Activity Separately")
    # Initialize Decision Tree Classifier
    DT = DecisionTreeClassifier(**DT_PARAMS)
    classify_activity_directly(DT, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_svm_to_classify_activity_directly(X_train, X_test, y_train, y_test):
    print("\n\nSVM - Classify Each Activity Separately")
    # Initialize the SVM classifier with the Gaussian Kernel
    SVM = svm.SVC(**SVM_PARAMS)
    classify_activity_directly(SVM, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_crf_to_classify_activity_directly(X_train, X_test, y_train, y_test):
    print("\n\nCRF - Classify Each Activity Separately")
    # Initialize CRF Classifier
    CRF_CLASSIFIER = CRF(**CRF_PARAMS)
    classify_activity_directly(CRF_CLASSIFIER, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_hmm_to_classify_activity_directly(X_train, X_test, y_train, y_test):
    print("\n\nHMM - Classify Activity Directly")
    # Initialize HMM Classifier
    HMM = hmm.GaussianHMM(
        n_components=y_train["activity"].nunique(), covariance_type="full", n_iter=100, random_state=42
    )
    classify_activity_directly(HMM, X_train, X_test, y_train, y_test, mode="probabilistic")


def classify_location_then_activity(model, X_train, X_test, y_train, y_test, mode="statistical"):
    """
    ##########################################
    Location then Activity
    ##########################################
    """
    print("\n\nClassify Location then Activity")

    # Select features related to presence in different locations
    X_train_location = X_train[
        ["bedroom_presence", "kitchen_presence", "bathroom_presence", "livingroom_presence_table"]
    ]
    X_test_location = X_test[["bedroom_presence", "kitchen_presence", "bathroom_presence", "livingroom_presence_table"]]

    if mode == "probabilistic":
        HMM_LOCATION = hmm.GaussianHMM(
            n_components=int(y_train["location"].nunique()), covariance_type="full", n_iter=100, random_state=42
        )
        # Train and evaluate the model for location prediction
        _, evaluation_results, location = model_train_test_score(
            HMM_LOCATION,
            X_train_location,
            X_test_location,
            y_train["location"],
            y_test["location"],
            LOCATIONS_LIST,
            mode=mode,
        )
    else:
        # Train and evaluate the model for location prediction
        _, evaluation_results, location = model_train_test_score(
            model,
            X_train_location,
            X_test_location,
            y_train["location"],
            y_test["location"],
            LOCATIONS_LIST,
            mode=mode,
        )

    print("Location Classification Results")
    print_metrices(evaluation_results)

    # Create a copy of the testing data for activity prediction and add location_int column
    X_test_activity = X_test.copy()
    X_test_activity["location_int"] = [LOCATION_INT_MAPPING[key] for key in location]

    # Create a copy of the training data for activity prediction and add location_int column
    X_train_activity = X_train.copy()
    X_train_activity["location_int"] = y_train["location_int"]

    # Get top features based on importance for activity prediction
    top_features = get_top_features_using_RFECV(X_train_activity, y_train["activity"])
    X_train_activity_subset = X_train_activity[top_features]
    X_test_activity_subset = X_test_activity[top_features]

    # Train and evaluate the model for activity prediction
    _, evaluation_results, _ = model_train_test_score(
        model,
        X_train_activity_subset,
        X_test_activity_subset,
        y_train["activity"],
        y_test["activity"],
        ACTIVITIES_LIST,
        mode=mode,
    )

    print("Activity Classification Results")
    print_metrices(evaluation_results)


@validate_experiment
def apply_decision_tree_to_classify_location_then_activity(X_train, X_test, y_train, y_test):
    print("\n\nDECISION TREE - Classify Each Activity Separately")
    # Initialize Decision Tree Classifier
    DT = DecisionTreeClassifier(**DT_PARAMS)
    classify_location_then_activity(DT, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_svm_to_classify_location_then_activity(X_train, X_test, y_train, y_test):
    print("\n\nSVM - Classify Each Activity Separately")
    # Initialize the SVM classifier with the Gaussian Kernel
    SVM = svm.SVC(**SVM_PARAMS)
    classify_location_then_activity(SVM, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_crf_to_classify_location_then_activity(X_train, X_test, y_train, y_test):
    print("\n\nCRF - Classify Each Activity Separately")
    # Initialize CRF Classifier
    CRF_CLASSIFIER = CRF(**CRF_PARAMS)
    classify_location_then_activity(CRF_CLASSIFIER, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_hmm_to_classify_location_then_activity(X_train, X_test, y_train, y_test):
    print("\n\nHMM - Classify Each Activity Separately")
    # Initialize HMM Classifier
    HMM = hmm.GaussianHMM(
        n_components=int(y_train["activity"].nunique()), covariance_type="full", n_iter=100, random_state=42
    )
    classify_location_then_activity(HMM, X_train, X_test, y_train, y_test, mode="probabilistic")


def apply_multiple_binary_classifiers_per_activity(model, X_train, X_test, y_train, y_test, mode="statistical"):
    # Create copies of training and testing data
    X_train_activity = X_train.copy()
    X_test_activity = X_test.copy()

    # Iterate over activities
    for activity in ACTIVITIES_LIST[:-1]:

        # Filter labels for the current activity
        y_train_activity = y_train["activity"].apply(lambda x: x if x == activity else "unknown")
        y_test_activity = y_test["activity"].apply(lambda x: x if x == activity else "unknown")

        # Select features related to the room of the current activity
        room_features = [col for col in X_train_activity.columns if col.startswith(ACTIVITY_LOCATION_MAPPING[activity])]

        # Get top features based on importance
        top_features = get_top_features_using_RFECV(X_train_activity[room_features], y_train_activity)
        X_train_activity_subset = X_train_activity[top_features]
        X_test_activity_subset = X_test_activity[top_features]

        # Train and evaluate the model
        _, evaluation_results, _ = model_train_test_score(
            model,
            X_train_activity_subset,
            X_test_activity_subset,
            y_train_activity,
            y_test_activity,
            [activity, "unknown"],
            mode=mode,
        )

        # Print evaluation metrics for the current activity
        print("Classification Report :", activity)
        print_metrices(evaluation_results)


@validate_experiment
def apply_multiple_binary_decision_tree_classifiers_per_activity(X_train, X_test, y_train, y_test):
    print("\n\nDECISION TREE - Classify Each Activity Separately")
    # Initialize Decision Tree Classifier
    DT = DecisionTreeClassifier(**DT_PARAMS)
    apply_multiple_binary_classifiers_per_activity(DT, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_multiple_binary_svm_classifiers_per_activity(X_train, X_test, y_train, y_test):
    print("\n\nSVM - Classify Each Activity Separately")
    # Initialize the SVM classifier with the Gaussian Kernel
    SVM = svm.SVC(**SVM_PARAMS)
    apply_multiple_binary_classifiers_per_activity(SVM, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_multiple_binary_crf_classifiers_per_activity(X_train, X_test, y_train, y_test):
    print("\n\nCRF - Classify Each Activity Separately")
    # Initialize CRF Classifier
    CRF_CLASSIFIER = CRF(**CRF_PARAMS)
    apply_multiple_binary_classifiers_per_activity(CRF_CLASSIFIER, X_train, X_test, y_train, y_test)


@validate_experiment
def apply_multiple_binary_supervised_hmm_classifiers_per_activity(X_train, X_test, y_train, y_test):
    print("\n\nHMM - Classify Each Activity Separately")
    # Initialize HMM Classifier
    HMM = hmm.GaussianHMM(
        n_components=y_train["activity"].nunique(), covariance_type="full", n_iter=100, random_state=42
    )
    apply_multiple_binary_classifiers_per_activity(HMM, X_train, X_test, y_train, y_test, mode="probabilistic")


@validate_experiment
def apply_rule_based_system_to_classify_location_then_activity(X_test, y_test):
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
    print("Rule Based System - Location Classification Results")
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
    evaluation_results = calculate_metrics(
        y_test_rules["activity"], y_test_rules["activity_prediction"], ACTIVITIES_LIST
    )

    # Print the evaluation results for the activity prediction
    print("\n\n")
    print("Rule Based System - Activity Classification Results")
    print_metrices(evaluation_results)
