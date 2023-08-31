#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np

# User
from .base.base_mlreg import BaseMLRegressor
from .targeting import LassoTargeter
from .utils.preprocess import Preprocessor, PolynomialExpander

#TODO: Implement Preprocessing and PolynomialExpander

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
                 target_regressors=False,
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
                 **kwargs
                 ):
        # Initiate
        self.estimator = estimator
        self.target_regressors = target_regressors
        self.param_grid = param_grid
        self.cv_params = cv_params
        self.fold_type = fold_type
        self.n_cv_folds = n_cv_folds
        self.shuffle = shuffle
        self.test_size = test_size
        self.max_n_models = max_n_models
        self.n_cf_folds = n_cf_folds
        self.verbose = verbose
        
        # Fix
        self._fix_target_estimator()
        
        if self.target_regressors:
            # Instantiate targeter
            self.targeter = LassoTargeter(**kwargs)
            
        super().__init__(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv_params=self.cv_params,
            fold_type=self.fold_type,
            n_cv_folds=self.n_cv_folds,
            shuffle=self.shuffle,
            test_size=self.test_size,
            max_n_models=self.max_n_models,
            n_cf_folds=self.n_cf_folds,
            verbose=self.verbose)
        
    # -------------------------------------------------------------------------
    # Privat functions
    # -------------------------------------------------------------------------
    def _fix_target_estimator(self):
        if isinstance(self.estimator, str):
            if self.estimator.startswith("Target"):
                self.estimator = self.estimator.split("Target", maxsplit=1)[-1]
                self.target_regressors = True
            
    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------        
    def fit(self,X,y):
        
        # Break link
        X = X.copy()
        
        self.X_shape_before_targeting_ = X.shape
        
        if self.target_regressors:
            X = self.targeter.fit_transform(X=X,y=y)

            self.X_shape_after_targeting_ = X.shape
            
            if self.verbose:
                print(f"Dimension of original regressors: {self.X_shape_before_targeting_}")
                print(f"Dimension of targeted regressors: {self.X_shape_after_targeting_}")
            
        return super().fit(X=X,y=y)
        
    def predict(self,X):
                
        # Break link
        X = X.copy()
        
        if self.target_regressors:
            X = self.targeter.transform(X=X)
            
            if not X.shape[1]==self.X_shape_after_targeting_[1]:
                raise Exception(f"X used for testing (={X.shape[1]}) does not have the same dimension as X used for training (={self.X_shape_after_targeting_[1]})")
    
        return super().predict(X=X)
        









        
        
