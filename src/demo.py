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
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# This library
from mlregression.mlreg import MLRegressor
from mlregression.mlreg import RF
from mlregression.estimator.boosting import XGBRegressor, LGBMegressor
from mlregression.estimator import boosting
#------------------------------------------------------------------------------
# Data
#------------------------------------------------------------------------------
# Generate data
X, y = make_regression(n_samples=500,
                       n_features=10, 
                       n_informative=5,
                       n_targets=1,
                       bias=0.0,
                       coef=False,
                       random_state=1991)

X_train, X_test, y_train, y_test = train_test_split(X, y)

#------------------------------------------------------------------------------
# Example 1: Main use of MLRegressor
#------------------------------------------------------------------------------
# Instantiate model and specify the underlying regressor by a string
mlreg = MLRegressor(estimator="RandomForestRegressor",
                    max_n_models=2)

# Fit
mlreg.fit(X=X_train, y=y_train)

# Predict
y_hat = mlreg.predict(X=X_test)

# Access all the usual attributes
mlreg.best_score_
mlreg.best_estimator_

# Compute the score
mlreg.score(X=X_test,y=y_test)

#------------------------------------------------------------------------------
# Example 2: RF
#------------------------------------------------------------------------------
# Instantiate model
rf = RF(max_n_models=2)

# Fit
rf.fit(X=X_train, y=y_train)

# Predict and score
rf.score(X=X_test, y=y_test)

#------------------------------------------------------------------------------
# Example 3: XGBoost
#------------------------------------------------------------------------------
# Instantiate model
xgb = MLRegressor(estimator=XGBRegressor(),
                  max_n_models=2)

# Fit
xgb.fit(X=X_train, y=y_train)

# Predict and score
xgb.score(X=X_test, y=y_test)

#------------------------------------------------------------------------------
# Example 4: LightGBM
#------------------------------------------------------------------------------
# Instantiate model
lgbm = MLRegressor(estimator="LGBMegressor",
                  max_n_models=2)

# Fit
lgbm.fit(X=X_train, y=y_train)

# Predict and score
lgbm.score(X=X_test, y=y_test)

#------------------------------------------------------------------------------
# Example 5: Neural Nets
#------------------------------------------------------------------------------
# Instantiate model
nn = MLRegressor(estimator="MLPRegressor",
                  max_n_models=2)

# Fit
nn.fit(X=X_train, y=y_train)

# Predict and score
nn.score(X=X_test, y=y_test)

#------------------------------------------------------------------------------
# Example 6: LassoCV/RidgeCV/ElasticNetCV/LarsCV/LassoLarsCV (native scikit-learn implementation)
#------------------------------------------------------------------------------
# Instantiate model
penalized = MLRegressor(estimator="LassoCV")

# Fit
penalized.fit(X=X_train, y=y_train)

# Predict and score
penalized.score(X=X_test, y=y_test)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
from sklearn.linear_model import RidgeCV
estimator = RidgeCV()




# estimator_name="LGBMegressor"









