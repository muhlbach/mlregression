
# *** ATTENTION ***
Don't immidiately run `pip install mlregression`. See Section _Installation_.

# Machine learning regression (mlregression)

Machine Learning Regression (mlregrresion) is an off-the-shelf implementation of the most popular ML methods that automatically takes care of fitting and parameter tuning.

Currently, the __fully__ implemented models include:
- Ensemble trees (Random forests, XGBoost, LightGBM, GradientBoostingRegressor, ExtraTreesRegressor)
- Penalized regression (Ridge, Lasso, ElasticNet, Lars, LassoLars) 
- Neural nets (Simple neural nets with 1-5 hidden layers, rely activation, and early stopping)

_NB!_ When using penalized regressions, consider using the native CV-implementation from scikit-learn for speed. See Example 6 below.

In addition, all scikit-learn regressors can be supplied (e.g., LinearRegression, HuberRegressor, or BayesianRidge), but then one has to provide a parameter grid as well!

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
- xgboost >=1.3
- lightgbm >=3.2


## Installation
Before calling `pip install mlregression`, we recommend using `conda` to install the dependencies. In our experience, calling the following command works like a charm:
```
conda install -c conda-forge numpy">=1.19" pandas">=1.3" scikit-learn">=1" scikit-learn-intelex">=2021.3" daal">=2021.3" daal4py">=2021.3" tbb">=2021.4" xgboost">=1.3" lightgbm">=3.2" --force-reinstall
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
from mlregression.mlreg import RF
from mlregression.estimator.boosting import XGBRegressor, LGBMegressor

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
# Example 2: Linear regression
#------------------------------------------------------------------------------
# Instantiate model
ols = MLRegressor(estimator="LinearRegression")

# Fit
ols.fit(X=X_train, y=y_train)

# Predict and score
ols.score(X=X_test, y=y_test)

#------------------------------------------------------------------------------
# Example 3: XGBoost
#------------------------------------------------------------------------------
# Instantiate model
xgb = MLRegressor(estimator="XGBRegressor",
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
# Example 6: LassoCV/RidgeCV/ElasticNetCV (native scikit-learn implementation)
#------------------------------------------------------------------------------
# Instantiate model
penalized = MLRegressor(estimator="LassoCV")

# Fit
penalized.fit(X=X_train, y=y_train)

# Predict and score
penalized.score(X=X_test, y=y_test)
```

<!-- ## Example
We provide an example script in `demo.py`. -->
