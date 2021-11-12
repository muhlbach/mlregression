#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit

class A():
    pass

self = A()

self.estimator = RandomForestRegressor()

#------------------------------------------------------------------------------
# MLRegressor
#------------------------------------------------------------------------------
# class MLRegressor():
#     """
#     This class implements the mlreg command
#     """
#     # --------------------
#     # Constructor function
#     # --------------------
#     # HERE!
#     def __init__(self,
#                  estimator=ConstrainedOLS(),
#                  param_grid={'coefs_lower_bound':0,
#                              'coefs_lower_bound_constraint':">=",
#                              'coefs_sum_bound':1,
#                              'coefs_sum_bound_constraint':"<=",},
#                  cv_params={'scoring':None,
#                             'n_jobs':None,
#                             'refit':True,
#                             'verbose':0,
#                             'pre_dispatch':'2*n_jobs',
#                             'random_state':None,
#                             'error_score':np.nan,
#                             'return_train_score':False},
#                  n_folds=3,
#                  fold_type="KFold",
#                  max_n_models=50,
#                  test_size=0.25,
#                  verbose=False,
#                  ):
#         super().__init__(
#             estimator=estimator,
#             param_grid=param_grid,
#             cv_params=cv_params,
#             n_folds=n_folds,
#             fold_type=fold_type,
#             max_n_models=max_n_models,
#             test_size=test_size,
#             verbose=verbose,
#             )

