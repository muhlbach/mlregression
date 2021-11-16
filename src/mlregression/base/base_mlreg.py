#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit


# User
from ..utils.tools import SingleSplit, remove_conditionally_invalid_keys,unlist_dict_values
from ..utils.model_params import get_param_grid_from_estimator, compose_model, update_params
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
            self.estimator = compose_model(estimator_name=self.estimator,
                                           perform_estimator_check=True,
                                           verbose=self.verbose)
        else:
            check_estimator(self.estimator)
        
        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------
        # Obtain parameters if not provided
        if self.param_grid is None:
            self.param_grid = get_param_grid_from_estimator(estimator=self.estimator)
        
        # Add default parameters if for some reason not specified    
        self.param_grid = update_params(old_param=self.estimator.get_params(),
                                        new_param=self.param_grid)
        
        # Remove invalid keys
        self.param_grid = remove_conditionally_invalid_keys(d=self.param_grid,
                                                            invalid_values=["deprecated"])
        
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
                                                   param_grid=self.param_grid,
                                                   cv_params=self.cv_params,                                                   
                                                   splitter=self.splitter,
                                                   n_models=self.n_models,
                                                   max_n_models=self.max_n_models)
           

    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    FOLD_TYPE_ALLOWED = ["KFold", "TimeSeriesSplit"]
    N_FOLDS_ALLOWED = [1, 2, "...", "N"]

    # -------------------------------------------------------------------------
    # Very Private functions
    # -------------------------------------------------------------------------
    def __getattr__(self, attrname):
        if not attrname in self.__dict__:
            return getattr(self.estimator_cv, attrname)

    # -------------------------------------------------------------------------
    # Private functions
    # -------------------------------------------------------------------------
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
        
    def _choose_estimator(self, estimator, param_grid, cv_params, splitter, n_models, max_n_models):
        """
        Choose between grid search or randomized search, or simply the estimator if only one parametrization is provided.
        Note that if the estimator ends with "CV", we assume it has a native implementation as in scikit-learn. We use that in this case.
        """       
        if type(estimator).__name__[-2:]=="CV":
            # Unlist if possible
            param_grid = unlist_dict_values(d=param_grid)
            
            # Get first element of parameter grid
            # param_grid = {k: param_grid.get(k,None)[0] for k in param_grid.keys()}
            # Set estimator
            estimator_cv = estimator
                        
            # Update params
            param_grid = update_params(old_param=param_grid,
                                       new_param={
                                           "cv":splitter
                                           }
                                       )
            
            # Set parameters
            estimator_cv.set_params(**param_grid)
            
        else:
            if n_models>1:            
                if n_models>max_n_models:
                    estimator_cv = RandomizedSearchCV(estimator=estimator,
                                                      param_distributions=param_grid,
                                                      cv=splitter,
                                                      n_iter=max_n_models,
                                                      **cv_params)
                else:
                    estimator_cv = GridSearchCV(estimator=estimator,
                                                param_grid=param_grid,
                                                cv=splitter,
                                                **cv_params)
            
            else:
                # If param_grid leads to one single model (n_models==1), there's no need to set of cross validation. In this case, just initialize the model and set parameters
                estimator_cv = estimator
                param_grid = {k: param_grid.get(k,None)[0] for k in param_grid.keys()}
                
                # Set parameters
                estimator_cv.set_params(**param_grid)        
            
        return estimator_cv
        
    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def fit(self,X,y):
        
        # Check X and Y
        X, y = check_X_Y(X, y)
                        
        # Estimate f in Y0 = f(X) + eps
        self.estimator_cv.fit(X=X,y=y)
        
        # Mean cross-validated score of the best_estimator
        if hasattr(self.estimator_cv, "best_score_"):     
            self.best_mean_cv_score_ = self.estimator_cv.best_score_
        else:
            self.best_mean_cv_score_ = None
                
        return self

    def get_params(self,deep=True):            
        return self.estimator_cv.get_params(deep=deep)

    def predict(self,X):
        
        # Check X and Y
        X = check_X(X)
                        
        # Estimate f in Y0 = f(X) + eps
        y_hat = self.estimator_cv.predict(X=X)
                        
        return y_hat
        
    def score(self,X,y):
        
        # Check X and Y
        X, y = check_X_Y(X, y)
                        
        score = self.estimator_cv.score(X=X,y=y)
                
        return score

    def set_params(self,**params):
        self.estimator_cv.set_params(**params)

    
    
    
    
    
    
    
    
    
        