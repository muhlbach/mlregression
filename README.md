
# Machine learning regression (mlregression)

Machine Learning Regression (mlregrresion) is an off-the-shelf implementation fitting and tuning the most popular ML methods (provided by scikit-learn)

Additionally, please contact the authors below if you find any bugs or have any suggestions for improvement. Thank you!

Author: Nicolaj Søndergaard Mühlbach (n.muhlbach at gmail dot com, muhlbach at mit dot edu) 

## Code dependencies
This code has the following dependencies:

- Python 3.6+
- numpy 1.19+
- pandas 1.3+
- scikit-learn 1+
- xgboost 1.3+
- lightgbm 3.2+

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
# Main use of MLRegressor
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
# RF
#------------------------------------------------------------------------------
# Instantiate model
rf = RF(max_n_models=2)

# Fit
rf.fit(X=X_train, y=y_train)

# Predict and score
rf.score(X=X_test, y=y_test)

#------------------------------------------------------------------------------
# XGBoost
#------------------------------------------------------------------------------
# Instantiate model
xgb = MLRegressor(estimator=XGBRegressor(),
                  max_n_models=2)

# Fit
xgb.fit(X=X_train, y=y_train)

# Predict and score
xgb.score(X=X_test, y=y_test)

#------------------------------------------------------------------------------
# LightGBM
#------------------------------------------------------------------------------
# Instantiate model
lgbm = MLRegressor(estimator=LGBMegressor(),
                  max_n_models=2)

# Fit
lgbm.fit(X=X_train, y=y_train)

# Predict and score
lgbm.score(X=X_test, y=y_test)
```

<!-- ## Example
We provide an example script in `demo.py`. -->
