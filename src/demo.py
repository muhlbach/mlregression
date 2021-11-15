#------------------------------------------------------------------------------
# BETA
#------------------------------------------------------------------------------
# import os
# # Manually set path of current file
# path_to_here = "/Users/muhlbach/Repositories/mlregression/src"
# # Change path
# os.chdir(path_to_here)

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from mlregression.base.base_mlreg import BaseMLRegressor


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
# Specify estimator
estimator = "RandomForestRegressor"

# Generate data
X, y = make_regression(n_samples=500,
                       n_features=10, 
                       n_informative=5,
                       n_targets=1,
                       bias=0.0,
                       coef=False,
                       random_state=1991)

# Instantiate model
rf = BaseMLRegressor(estimator=estimator,
                     max_n_models=2)

# Fit
rf.fit(X=X, y=y)

# Access all the usual attributes
rf.best_score_
rf.best_estimator_



