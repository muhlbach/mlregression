
# *** ATTENTION ***
Don't immidiately run `pip install mlregression`. See Section _Installation_.

# Machine learning regression (mlregression)

Machine Learning Regression (mlregrresion) is an off-the-shelf implementation of the most popular ML methods that automatically takes care of fitting and parameter tuning.

Currently, the __fully__ implemented models include:
- Ensemble trees (Random forests, XGBoost, LightGBM, GradientBoostingRegressor, ExtraTreesRegressor)
- Penalized regression (Ridge, Lasso, ElasticNet, Lars, LassoLars) 
- Neural nets (Simple neural nets with 1-5 hidden layers, rely activation, and early stopping)

_NB!_ When using penalized regressions, consider using the native CV-implementation from scikit-learn for speed, e.g., simply set `estimator="LassoCV"` similar to Example 1.

Scikit-learn regressors (together with `XGBoost` and `LightGBM`) can be estimated by setting the `estimator`-argument equal to the name (string) as in Example 1 (`estimator="RandomForestRegressor"`).
Alternatively, one can provide an instance of an estimator, e.g., `estimator=RandomForestRegressor()`. Again, this is fully automated for most Scikit-learn regressors, but for non-standard methods, one would have to provide a parameter grid as well, e.g., `param_grid={...}`.

Please contact the authors below if you find any bugs or have any suggestions for improvement. Thank you!

Author: Nicolaj Søndergaard Mühlbach (n.muhlbach at gmail dot com, muhlbach at mit dot edu) 

## Code dependencies
This code has the following dependencies:

- Python >=3.6
- numpy >=1.19
- pandas >=1.3
- scikit-learn >=1
- scikit-learn-intelex >= 2021.3
- daal >= 2021.3
- daal4py >= 2021.3
- tbb >= 2021.4
- xgboost >=1.5
- lightgbm >=3.2


## Installation
Before calling `pip install mlregression`, we recommend using `conda` to install the dependencies. In our experience, calling the following command works like a charm:
```
conda install -c conda-forge numpy">=1.19" pandas">=1.3" scikit-learn">=1" scikit-learn-intelex">=2021.3" daal">=2021.3" daal4py">=2021.3" tbb">=2021.4" xgboost">=1.5" lightgbm">=3.2" --force-reinstall
```
After this, install `mlregression` by calling `pip install mlregression`.
Note that without installing the dependensies, the package will not work. As of now, it does not work when installing the dependensies via `pip install`. The reason is that we are using the Intel® Extension for Scikit-learn to massively speed up computations, but the dependensies are not properly installed via `pip install`.

## Usage
We demonstrate the use of __mlregression__ below, using random forests, xgboost, and lightGBM as underlying regressors.

```python
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
```

<!-- ## Example
We provide an example script in `demo.py`. -->
