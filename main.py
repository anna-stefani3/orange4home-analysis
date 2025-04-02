"""
Install required modules using command inside quotes

`pip install pandas numpy scikit-learn sklearn-crfsuite imbalanced-learn hmmlearn`
"""

from preprocessing import apply_preprocessing
from common_variables import BASE_PATH
from experiments import (
    apply_rule_based_system_to_classify_location_then_activity,
    apply_decision_tree_to_classify_location_then_activity,
    apply_random_forest_to_classify_location_then_activity,
    apply_svm_to_classify_location_then_activity,
    apply_hmm_to_classify_activity_directly,
    apply_decision_tree_to_classify_activity_directly,
    apply_random_forest_to_classify_activity_directly,
    apply_svm_to_classify_activity_directly,
    apply_hmm_to_classify_location_then_activity,
)

from preprocessing import get_balanced_training_dataset

# get train, valid, test splitted data
X_train, X_valid, X_test, y_train, y_valid, y_test = apply_preprocessing(filename=BASE_PATH)
X_train_balanced, y_train_balanced = get_balanced_training_dataset(X_train, y_train)


# BALANCED DATA
apply_decision_tree_to_classify_activity_directly(X_train_balanced, X_valid, y_train_balanced, y_valid)
apply_random_forest_to_classify_activity_directly(X_train_balanced, X_valid, y_train_balanced, y_valid)
apply_svm_to_classify_activity_directly(X_train_balanced, X_valid, y_train_balanced, y_valid)
apply_hmm_to_classify_activity_directly(X_train_balanced, X_valid, y_train_balanced, y_valid)


# ACTIVITY DIRECTLY
apply_decision_tree_to_classify_activity_directly(X_train, X_valid, y_train, y_valid)
apply_random_forest_to_classify_activity_directly(X_train, X_valid, y_train, y_valid)
apply_svm_to_classify_activity_directly(X_train, X_valid, y_train, y_valid)
apply_hmm_to_classify_activity_directly(X_train, X_valid, y_train, y_valid)


# LOCATION THEN ACTIVITY
apply_decision_tree_to_classify_location_then_activity(X_train, X_valid, y_train, y_valid)
apply_random_forest_to_classify_location_then_activity(X_train, X_valid, y_train, y_valid)
apply_svm_to_classify_location_then_activity(X_train, X_valid, y_train, y_valid)
apply_hmm_to_classify_location_then_activity(X_train, X_valid, y_train, y_valid)
apply_rule_based_system_to_classify_location_then_activity(
    X_valid, y_valid
)  # Rule based system only for Location then Activity


# TOP MODELS COMPARISON - USING TESTING DATA
apply_decision_tree_to_classify_location_then_activity(X_train, X_test, y_train, y_test)
apply_rule_based_system_to_classify_location_then_activity(X_test, y_test)
apply_svm_to_classify_location_then_activity(X_train, X_test, y_train, y_test)
apply_hmm_to_classify_location_then_activity(X_train, X_test, y_train, y_test)
