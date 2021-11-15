
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

## Usage

```python
# Import
from sklearn.datasets import make_regression
from mlregression.base.base_mlreg import BaseMLRegressor

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
mlreg = BaseMLRegressor(estimator=estimator,
                        max_n_models=2)

# Fit
mlreg.fit(X=X, y=y)

# Access all the usual attributes
mlreg.best_score_
mlreg.best_estimator_
```

<!-- ## Example
We provide an example script in `demo.py`. -->
