#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# This library
from mlregression.mlreg import MLRegressor

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
# Example 1: Prediction
#------------------------------------------------------------------------------
# Specify any of the following estimators:
"""
"LinearRegression",
"RidgeCV", "LassoCV", "ElasticNetCV",
"RandomForestRegressor","ExtraTreesRegressor", "GradientBoostingRegressor",
"XGBRegressor", "LGBMegressor",
"MLPRegressor",
"""

# For instance, pick "RandomForestRegressor"
estimator = "RandomForestRegressor"
# Note that the 'estimator' may also be an instance of a class, e.g., RandomForestRegressor(), conditional on being imported first, e.g. from sklearn.ensemble import RandomForestRegressor

# Instantiate model and choose the number of parametrizations to examine using cross-validation ('max_n_models') and the number of cross-validation folds ('n_cv_folds')
mlreg = MLRegressor(estimator=estimator,
                    n_cv_folds=5,
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
# Example 2: Cross-fitting
#------------------------------------------------------------------------------
# Instantiate model and choose the number of parametrizations to examine using cross-validation ('max_n_models'), the number of cross-validation folds ('n_cv_folds'), AND the number of cross-fitting folds ('n_cf_folds')
mlreg = MLRegressor(estimator=estimator,
                    n_cv_folds=5,
                    max_n_models=2,
                    n_cf_folds=2)

# Cross fit
mlreg.cross_fit(X=X_train, y=y_train)

# Extract in-sample that are estimated in an out-of-sample way (e.g., via cross-fitting)
y_hat = mlreg.y_pred_cf_

# Likewise, extract the residualized outcomes used in e.g., double machine learning. This is \tilde{Y} = Y - E[Y|X=x]
y_res = mlreg.y_res_cf_