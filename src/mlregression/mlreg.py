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
                            'error_score':np.nan,
                            'return_train_score':False},
                 fold_type="KFold",
                 n_cv_folds=5,
                 shuffle=False,
                 test_size=None,
                 max_n_models=50,
                 n_cf_folds=2,
                 verbose=False,
                 ):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            cv_params=cv_params,
            fold_type=fold_type,
            n_cv_folds=n_cv_folds,
            shuffle=shuffle,
            test_size=test_size,
            max_n_models=max_n_models,
            n_cf_folds=n_cf_folds,
            verbose=verbose)
