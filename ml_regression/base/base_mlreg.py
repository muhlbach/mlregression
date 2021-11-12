#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit

# User
from ..utils.tools import SingleSplit, get_param_grid_from_estimator
from ..utils.params import get_param_grid_from_estimator
from ..utils.sanity_check import check_param_grid, check_X_Y, check_X, check_estimator
from ..utils.exceptions import WrongInputException

#------------------------------------------------------------------------------
# Base ML Regressor
#------------------------------------------------------------------------------
class BaseMLRegressor(object):
    """
    This base class learns a conditional expectation of Y given X=x
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
        # Initialize inputs
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv_params = cv_params
        self.n_folds = n_folds
        self.test_size = test_size
        self.fold_type = fold_type
        self.max_n_models = max_n_models
        self.verbose = verbose

        # ---------------------------------------------------------------------
        # Estimator
        # ---------------------------------------------------------------------
        # Check estimator type
        if isinstance(self.estimator, str):
            # self.estimator = self._instantiate_estimator_from_string(name=self.estimator)
            raise NotImplementedError("Provided 'estimator' cannot be a string at this point, my apologies...")
        else:
            check_estimator(self.estimator)
            
        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------
        # Obtain parameters if not provided
        if self.param_grid is None:
            self.param_grid = get_param_grid_from_estimator(estimator=self.estimator)

        # Set param_grid values to list if not already list
        self.param_grid = {k: list(set(v)) if isinstance(v, list) else v.tolist() if isinstance(v, np.ndarray) else [v] for k, v in self.param_grid.items()}

        # Check parameter grid
        check_param_grid(self.param_grid)

        # ---------------------------------------------------------------------
        # Misc
        # ---------------------------------------------------------------------
        # Compute number of models
        self.n_models = np.prod(np.array([len(v) for k,v in self.param_grid.items()]))

        # Define data splitter used in cross validation
        self.splitter = self._choose_splitter(n_folds=self.n_folds, fold_type=self.fold_type, test_size=self.test_size)
            
        # Define cross-validated estimator
        self.estimator_cv = self._choose_estimator(estimator=self.estimator,
                                                   splitter=self.splitter,
                                                   n_models=self.n_models,
                                                   max_n_models=self.max_n_models,
                                                   param_grid=self.param_grid)

    # --------------------
    # Class variables
    # --------------------
    FOLD_TYPE_ALLOWED = ["KFold", "TimeSeriesSplit"]
    N_FOLDS_ALLOWED = [1, 2, "...", "N"]

    # --------------------
    # Private functions
    # --------------------
    def _update_params(self, old_param, new_param, errors="raise"):
        """ Update 'old_param' with 'new_param'
        """
        # Copy old param
        updated_param = old_param.copy()
        
        for k,v in new_param.items():
            if k in old_param:
                updated_param[k] = v
            else:
                if errors=="raise":
                    raise Exception(f"Parameters {k} not recognized as a default parameter for this estimator")
                else:
                    pass
        return updated_param

    def _choose_splitter(self, n_folds=2, fold_type="KFold", test_size=0.25):
        """ Define the split function that splits the data for cross-validation"""
        if n_folds==1:
            if test_size is None:
                raise Exception("""
                                When 'n_folds==1', we automatically instantiate a SingleSplit.
                                This requires 'test_size' to be a float and not 'None'
                                """)                
            else:
                splitter = SingleSplit(test_size=test_size)
            
        elif n_folds>=2:
            if fold_type=="KFold":
                splitter = KFold(n_splits=n_folds, random_state=None, shuffle=False)
            elif fold_type=="TimeSeriesSplit":
                splitter = TimeSeriesSplit(n_splits=n_folds, max_train_size=None, test_size=None, gap=0)
            else:
                raise WrongInputException(input_name="fold_type",
                                          provided_input=fold_type,
                                          allowed_inputs=self.FOLD_TYPE_ALLOWED)
        else:
            raise WrongInputException(input_name="n_folds",
                                      provided_input=n_folds,
                                      allowed_inputs=self.N_FOLDS_ALLOWED)        
        
        return splitter
        
    def _choose_estimator(self, estimator, splitter, n_models, max_n_models, param_grid):
        """ Choose between grid search or randomized search, or simply the estimator if only one parametrization is provided """
        if n_models>1:
            if n_models>max_n_models:
                estimator_cv = RandomizedSearchCV(estimator=estimator,
                                                  param_distributions=param_grid,
                                                  cv=splitter,
                                                  n_iter=max_n_models)
            else:
                estimator_cv = GridSearchCV(estimator=estimator,
                                            param_grid=param_grid,
                                            cv=splitter)
        
        else:
            # If param_grid leads to one single model (n_models==1), there's no need to set of cross validation. In this case, just initialize the model and set parameters
            estimator_cv = estimator
            param_grid = {k: param_grid.get(k,None)[0] for k in param_grid.keys()}
            
            # Set parameters
            estimator_cv.set_params(**param_grid)        
            
        return estimator_cv
        
    # --------------------
    # Public functions
    # --------------------
    def fit(self,X,y):
        
        # Check X and Y
        X, y = check_X_Y(X, y)
                        
        # Estimate f in Y0 = f(X) + eps
        self.estimator_cv.fit(X=X,y=y)
        
        # Mean cross-validated score of the best_estimator
        self.best_mean_cv_score_ = self.estimator_cv.best_score_
                
        return self
        
    def predict(self,X):
        
        # Check X and Y
        X = check_X(X)
                        
        # Estimate f in Y0 = f(X) + eps
        y_hat = self.estimator_cv.predict(X=X)
                        
        return y_hat
        
    
    
    
    
    
    
    
    
    
    
        