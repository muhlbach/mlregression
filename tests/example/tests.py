"""
This script implements some basic tests of the package
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import baseline modules
from sklearn.linear_model import LinearRegression as LinearRegressionBase
from sklearn.linear_model import RidgeCV as RidgeCVBase
from sklearn.linear_model import LassoCV as LassoCVBase
from sklearn.linear_model import ElasticNetCV as ElasticNetCVBase
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressorBase
from sklearn.ensemble import ExtraTreesRegressor as ExtraTreesRegressorBase
from sklearn.ensemble import GradientBoostingRegressor as GradientBoostingRegressorBase
from xgboost import XGBRegressor as XGBRegressorBase
from lightgbm import LGBMRegressor as LGBMegressorBase
from sklearn.neural_network import MLPRegressor as MLPRegressorBase

# This library
from mlregression.mlreg import MLRegressor
from mlregression.estimator.boosting import XGBRegressor, LGBMegressor
from mlregression.estimator import boosting

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
# Set tolerance for MSE-error
mse_tol_pct = 600

# Number of samples
n_obs = 1000

# Number of max models to run
max_n_models = 3

# All estimators as strings
estimator_strings = [
    "LinearRegression",
    "RidgeCV", "LassoCV", "ElasticNetCV",
    "RandomForestRegressor","ExtraTreesRegressor", "GradientBoostingRegressor",
    "XGBRegressor", "LGBMegressor",
    "MLPRegressor",
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

estimator="RandomForestRegressor"
mse_strings = {}
for i,estimator in enumerate(estimator_strings):
    print(f"""\nTesting {estimator} ~ {i+1}/{len(estimator_strings)}""")
    
    #--------------------------------------------------------------------------
    # Estimate base model
    #--------------------------------------------------------------------------
    # Instantiate base model without any parameters (using default)
    base_estimator = eval(estimator+"Base()")

    # Fit
    base_estimator.fit(X=X_train, y=y_train)
    
    # Predict
    y_hat_base = base_estimator.predict(X_test)
    
    # Compute mse
    mse_hat_base = mean_squared_error(y_true=y_test,y_pred=y_hat_base)
    
    #--------------------------------------------------------------------------
    # Estimate CV model
    #--------------------------------------------------------------------------
    # Instantiate model
    mlreg = MLRegressor(estimator=estimator,
                        max_n_models=max_n_models)
    
    # Fit
    mlreg.fit(X=X_train, y=y_train)
      
    # Predict
    y_hat = mlreg.predict(X=X_test)

    # Compute mse
    mse_hat = mean_squared_error(y_true=y_test,y_pred=y_hat)
    
    # Compare
    if (mse_hat - mse_hat_base) > (mse_tol_pct*mse_hat_base):
        raise Exception(f"""
                        Out-of-sample performance of estimator '{estimator}' is more than {mse_tol_pct*100}% worse than base implementation!
                        MSE={round(mse_hat,4)} but baseline MSE={round(mse_hat_base,4)}
                        Check code!
                        """)
    
    # Store mse
    mse_strings[estimator] = mse_hat

    #--------------------------------------------------------------------------
    # Crossfitting
    #--------------------------------------------------------------------------
    # Fit
    mlreg.cross_fit(X=X_train, y=y_train)

    # Check res
    if not all(y_train-mlreg.y_pred_cf_ == mlreg.y_res_cf_):
        raise Exception(f"""
                        Cross-fitting went wrong for estimator '{estimator}'
                        Check code!
                        """)
                        
    # Compute mse
    mse_hat_ins = mean_squared_error(y_true=y_train,y_pred=mlreg.y_pred_cf_)

    # Compare
    if (mse_hat_ins - mse_hat_base) > (mse_tol_pct*mse_hat_base):
        raise Exception(f"""
                        In-sample performance of estimator '{estimator}' is more than {mse_tol_pct*100}% worse than base implementation!
                        MSE={round(mse_hat_ins,4)} but baseline MSE={round(mse_hat_base,4)}
                        Check code!
                        """)

    #--------------------------------------------------------------------------
    # The End
    #--------------------------------------------------------------------------
    print(f"""Succes of {estimator}!
          MSE(base) = {round(mse_hat_base,8)}
          MSE(out-of-sample) = {round(mse_hat,8)}
          MSE(cross-fitting in-sample) = {round(mse_hat_ins,8)}
          """)



















