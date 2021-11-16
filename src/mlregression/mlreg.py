#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np

# User
from .base.base_mlreg import BaseMLRegressor

#------------------------------------------------------------------------------
# MLRegressor
#------------------------------------------------------------------------------
class MLRegressor(BaseMLRegressor):
    """
    This class implements the mlreg command
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
               estimator,
               param_grid=None,
               cv_params={'scoring':None,
                          'n_jobs':None,
                          'refit':True,
                          'verbose':0,
                          'pre_dispatch':'2*n_jobs',
                          'random_state':None,
                          'error_score':np.nan,
                          'return_train_score':False},
               n_folds=3,
               test_size=None,
               fold_type="KFold",
               max_n_models=50,
               verbose=False,
               ):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            cv_params=cv_params,
            n_folds=n_folds,
            test_size=test_size,
            fold_type=fold_type,
            max_n_models=max_n_models,
            verbose=verbose)

#------------------------------------------------------------------------------
# RandomForest
#------------------------------------------------------------------------------
class RF(BaseMLRegressor):
    """
    This class implements the mlreg command
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self, 
               param_grid=None,
               cv_params={'scoring':None,
                          'n_jobs':None,
                          'refit':True,
                          'verbose':0,
                          'pre_dispatch':'2*n_jobs',
                          'random_state':None,
                          'error_score':np.nan,
                          'return_train_score':False},
               n_folds=3,
               test_size=None,
               fold_type="KFold",
               max_n_models=50,
               verbose=False,
               ):
        super().__init__(
            estimator="RandomForestRegressor",
            param_grid=param_grid,
            cv_params=cv_params,
            n_folds=n_folds,
            test_size=test_size,
            fold_type=fold_type,
            max_n_models=max_n_models,
            verbose=verbose)
