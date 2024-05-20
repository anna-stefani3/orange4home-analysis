"""
Install required modules using command inside quotes

`pip install pandas numpy scikit-learn imbalanced-learn`
"""

from preprocessing import apply_preprocessing
from common_variables import BASE_PATH
from experiments import (
    apply_rule_based_system,
    apply_method_1,
    apply_method_2,
    apply_method_3,
    apply_balanced_dt_method,
)

# get train, valid, test splitted data
X_train, X_valid, X_test, y_train, y_valid, y_test = apply_preprocessing(filename=BASE_PATH)


apply_balanced_dt_method(X_train, X_valid, y_train, y_valid)
apply_method_1(X_train, X_valid, y_train, y_valid)
apply_method_2(X_train, X_valid, y_train, y_valid)
apply_method_3(X_train, X_valid, y_train, y_valid)

apply_rule_based_system(X_valid, y_valid)
