"""
Install required modules using command inside quotes

`pip install pandas numpy scikit-learn imbalanced-learn`
"""

from preprocessing import apply_preprocessing
from common_variables import BASE_PATH
from experiments import (
    apply_rule_based_system,
    apply_decision_tree_to_classify_activity_directly,
    apply_decision_tree_to_classify_location_then_activity,
    apply_separate_decision_tree_for_each_activity,
    apply_balanced_dt_method,
)

# get train, valid, test splitted data
X_train, X_valid, X_test, y_train, y_valid, y_test = apply_preprocessing(filename=BASE_PATH)


apply_balanced_dt_method(X_train, X_valid, y_train, y_valid)
apply_decision_tree_to_classify_activity_directly(X_train, X_valid, y_train, y_valid)
apply_decision_tree_to_classify_location_then_activity(X_train, X_valid, y_train, y_valid)
apply_separate_decision_tree_for_each_activity(X_train, X_valid, y_train, y_valid)

apply_rule_based_system(X_valid, y_valid)
