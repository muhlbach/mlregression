"""
This script implements some basic tests of the package that should be run before uploading
"""
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
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# This library
from mlregression.mlreg import MLRegressor
from mlregression.mlreg import RF
from mlregression.estimator.boosting import XGBRegressor, LGBMegressor
from mlregression.estimator import boosting

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
# Set tolerance for MSE-error
mse_tol = 1.5

# Number of samples
n_obs = 400

# Number of max models to run
max_n_models = 50

# All estimators as strings
estimator_strings = ["RandomForestRegressor", "XGBRegressor", "LGBMegressor","ExtraTreesRegressor", "GradientBoostingRegressor",
                     "MLPRegressor",
                     "RidgeCV","LassoCV","ElasticNetCV",
                     ]

#------------------------------------------------------------------------------
# Data
#------------------------------------------------------------------------------
# Generate data
X, y = make_regression(n_samples=n_obs,
                       n_features=10, 
                       n_informative=5,
                       n_targets=1,
                       bias=0.0,
                       coef=False,
                       random_state=1991)

X_train, X_test, y_train, y_test = train_test_split(X, y)


#------------------------------------------------------------------------------
# Estimator as strings
#------------------------------------------------------------------------------
print("""
      *************************************************************************
      TESTING ESTIMATORS AS STRINGS
      *************************************************************************
      """)

# estimator="RandomForestRegressor"
mse_strings = {}
for i,estimator in enumerate(estimator_strings):
    print(f"""\nTesting {estimator} ~ {i+1}/{len(estimator_strings)}""")
    
    # Instantiate model
    mlreg = MLRegressor(estimator=estimator,
                        max_n_models=max_n_models)
    
    # Fit
    mlreg.fit(X=X_train, y=y_train)
    
    mlreg.best_params_
    
    # Predict
    y_hat = mlreg.predict(X=X_test)

    # Compute mse
    mse_hat = mean_squared_error(y_true=y_test,y_pred=y_hat)
    
    if mse_hat>mse_tol:
        raise Exception(f"""
                        Performance of estimator '{estimator}' worse than tolerated!
                        MSE={round(mse_hat,4)} but tolerance={mse_tol}
                        Check coude!
                        """)
    
    # Store mse
    mse_strings[estimator] = mse_hat

    print(f"""\tSuccesful! MSE({estimator}) = {round(mse_hat,8)}""")

# HERE: Compare to default model !!!!!




# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor()
# rf.set_params(**mlreg.best_params_)

# rf.fit(X=X_train, y=y_train)

# y_rf = rf.predict(X=X_test)

# mean_squared_error(y_true=y_test,y_pred=y_rf)


# #------------------------------------------------------------------------------
# # Example 1: Main use of MLRegressor
# #------------------------------------------------------------------------------
# # Instantiate model and specify the underlying regressor by a string
# mlreg = MLRegressor(estimator="RandomForestRegressor",
#                     max_n_models=2)

# # Fit
# mlreg.fit(X=X_train, y=y_train)

# # Predict
# y_hat = mlreg.predict(X=X_test)

# # Access all the usual attributes
# mlreg.best_score_
# mlreg.best_estimator_

# # Compute the score
# mlreg.score(X=X_test,y=y_test)

# #------------------------------------------------------------------------------
# # Example 2: RF
# #------------------------------------------------------------------------------
# # Instantiate model
# rf = RF(max_n_models=2)

# # Fit
# rf.fit(X=X_train, y=y_train)

# # Predict and score
# rf.score(X=X_test, y=y_test)

# #------------------------------------------------------------------------------
# # Example 3: XGBoost
# #------------------------------------------------------------------------------
# # Instantiate model
# xgb = MLRegressor(estimator=XGBRegressor(),
#                   max_n_models=2)

# # Fit
# xgb.fit(X=X_train, y=y_train)

# # Predict and score
# xgb.score(X=X_test, y=y_test)

# #------------------------------------------------------------------------------
# # Example 4: LightGBM
# #------------------------------------------------------------------------------
# # Instantiate model
# lgbm = MLRegressor(estimator="LGBMegressor",
#                   max_n_models=2)

# # Fit
# lgbm.fit(X=X_train, y=y_train)

# # Predict and score
# lgbm.score(X=X_test, y=y_test)

# #------------------------------------------------------------------------------
# # Example 5: Neural Nets
# #------------------------------------------------------------------------------
# # Instantiate model
# nn = MLRegressor(estimator="MLPRegressor",
#                   max_n_models=2)

# # Fit
# nn.fit(X=X_train, y=y_train)

# # Predict and score
# nn.score(X=X_test, y=y_test)

# #------------------------------------------------------------------------------
# # Example 6: LassoCV/RidgeCV/ElasticNetCV/LarsCV/LassoLarsCV (native scikit-learn implementation)
# #------------------------------------------------------------------------------
# # Instantiate model
# penalized = MLRegressor(estimator="LassoCV")

# # Fit
# penalized.fit(X=X_train, y=y_train)

# # Predict and score
# penalized.score(X=X_test, y=y_test)

# #------------------------------------------------------------------------------
# # Tests
# #------------------------------------------------------------------------------
# from sklearn.linear_model import LarsCV, Lars
# estimator = LarsCV()
# estimator = Lars()



# # estimator_name="LGBMegressor"









