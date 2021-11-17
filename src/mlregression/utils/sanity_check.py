#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.base import is_regressor
from collections.abc import Sequence
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def check_estimator(estimator):
    """Check estimator"""    
    if not is_regressor(estimator):
        raise Exception(f"Estimator '{type(estimator).__name__}' did not pass an a regressor")
    
    # Required methods of an regression model
    REQUIRED_METHODS = ["get_params", "set_params", "fit", "predict", "score"]

    # Check if class has required methods by using getattr() to get the attribute, and callable() to verify it is a method
    for method in REQUIRED_METHODS:
        method_attribute = getattr(estimator, method, None)
        if not callable(method_attribute):
            raise Exception(f"Estimator '{type(estimator).__name__}' does not contain the method '{method}'")

def check_param_grid(param_grid):
    if hasattr(param_grid, "items"):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if isinstance(v, str) or not isinstance(v, (np.ndarray, Sequence)):
                raise ValueError(
                    "Parameter grid for parameter ({0}) needs to"
                    " be a list or numpy array, but got ({1})."
                    " Single values need to be wrapped in a list"
                    " with one element.".format(name, type(v))
                )

            if len(v) == 0:
                raise ValueError(
                    "Parameter values for parameter ({0}) need "
                    "to be a non-empty sequence.".format(name)
                )
                
def check_X(X):
    
    # Convert to dataframe if not already converted
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Check for numeric dtype
    if not all(X.dtypes.map(pd.api.types.is_numeric_dtype)):
        raise Exception("X must be only numeric dtype")
    
    # Check for missing
    if X.isnull().values.any():
        raise Exception("X contains missing inputs")
        
    # Convert back to np.array
    X = np.array(X)
        
    return X
                
def check_Y(Y):
    
    # Convert to series if not already converted
    if not isinstance(Y, pd.Series):
        Y = pd.Series(Y)
    
    # Check for numeric dtype
    if not pd.api.types.is_numeric_dtype(Y.dtype):
        raise Exception("Y must be numeric dtype")
    
    # Check for missing
    if Y.isnull().values.any():
        raise Exception("Y contains missing inputs")        
    
    # Convert back to np.array
    Y = np.array(Y)    
    
    return Y
        
def check_X_Y(X, Y):
    
    # Perform individual checks
    X = check_X(X)
    Y = check_Y(Y)
    
    # Check length
    if len(Y)!=len(X):
        raise Exception(f"Y has {len(Y)} observations, whereas X has {len(X)} observations. They must match.")

    return X, Y