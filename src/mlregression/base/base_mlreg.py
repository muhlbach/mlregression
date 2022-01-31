#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
from sklearnex import patch_sklearn
patch_sklearn()
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, GridSearchCV, KFold, TimeSeriesSplit
from copy import deepcopy
import inspect

# User
from ..utils.tools import (SingleSplit,
                           isin,
                           remove_conditionally_invalid_keys,
                           unlist_dict_values, get_unique_elements_from_list)
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
                            'error_score':np.nan,
                            'return_train_score':False},
                 fold_type="KFold",
                 n_cv_folds=5,
                 shuffle=False,
                 test_size=None,
                 max_n_models=50,
                 n_cf_folds=None,
                 verbose=False,
                 ):
        # Initialize inputs
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv_params = cv_params
        self.n_cv_folds = n_cv_folds
        self.shuffle = shuffle
        self.test_size = test_size
        self.fold_type = fold_type
        self.max_n_models = max_n_models
        self.n_cf_folds = n_cf_folds
        self.verbose = verbose

        # ---------------------------------------------------------------------
        # Estimator
        # ---------------------------------------------------------------------
        """This instantiates the estimator as self.estimator"""
        
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
        self.param_grid = self._fix_params(estimator=self.estimator,
                                           param_grid=self.param_grid)

        # ---------------------------------------------------------------------
        # Misc
        # ---------------------------------------------------------------------
        # Define data splitter used in cross validation
        self.splitter = self._choose_splitter(n_folds=self.n_cv_folds,
                                              fold_type=self.fold_type,
                                              shuffle=self.shuffle,
                                              test_size=self.test_size)
            
        # Define cross-validated estimator
        self.estimator_cv = self._choose_estimator(estimator=self.estimator,
                                                   param_grid=self.param_grid,
                                                   cv_params=self.cv_params,                                                   
                                                   splitter=self.splitter,
                                                   max_n_models=self.max_n_models)
           

    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    FOLD_TYPE_ALLOWED = ["KFold", "TimeSeriesSplit", "SingleSplit"]
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
    def _fix_params(self, estimator, param_grid):
        """ Fix parameters, including get parameters if not provided, remove invalids, convert to list, and remove duplicates"""
        
        # Obtain default grid parameters if not provided
        if param_grid is None:
            param_grid = get_param_grid_from_estimator(estimator=estimator)
                        
        # print(f"\n\n\nparam within fnc1: {param_grid} \n\n")
        # Add default parameters if for some reason not specified by practitioner
        param_grid = update_params(old_param=estimator.get_params(),
                                   new_param=param_grid)
        
        # Obtain signature of composed estimator
        signature = inspect.signature(estimator.__init__)

        # Obtain parameters that was part of __init__
        init_params = list(signature.parameters.keys())
                
        # We require that all parameters are part of initialization
        if not isin(a=list(param_grid.keys()),
                    b=init_params,
                    how="all",
                    return_element_wise=False):
            raise Exception(f"""
                            \nSome parameters are not initialized. Check estimator!
                            \nInitial parameters: {init_params}
                            \nParameter grid: {list(param_grid.keys())}
                            \nMissing parameters: {[set(param_grid.keys())-set(init_params)]}
                            """)
                            
        # Remove invalid keys
        param_grid = remove_conditionally_invalid_keys(d=param_grid,
                                                        invalid_values=["deprecated"])
                
        # Set param_grid values to list if not already list
        param_grid = {k: v if isinstance(v, list) else v.tolist() if isinstance(v, np.ndarray) else [v] for k, v in param_grid.items()}

        # Remove duplicates
        param_grid = {k: get_unique_elements_from_list(l=v,keep_order=True) for k,v in param_grid.items()}
        
        # Check parameter grid
        check_param_grid(param_grid)
        
        return param_grid
        
    def _choose_splitter(self, n_folds=2, fold_type="KFold", shuffle=True, test_size=0.25):
        """ Define the split function that splits the data for cross-validation"""
        if (n_folds==1) or (fold_type=="SingleSplit"):
            if test_size is None:
                raise Exception("""
                                When 'n_folds==1', we automatically instantiate a SingleSplit.
                                This requires 'test_size' to be a float and not 'None'
                                """)                
            else:
                splitter = SingleSplit(test_size=test_size)
            
        elif n_folds>=2:
            if fold_type=="KFold":
                splitter = KFold(n_splits=n_folds, random_state=None, shuffle=shuffle)
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
        
    def _choose_estimator(self, estimator, param_grid, cv_params, splitter, max_n_models):
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
            # Compute number of models
            n_models = np.prod(np.array([len(v) for k,v in param_grid.items()]))
                        
            if n_models>1:            
                
                if n_models>max_n_models:
                    
                    # max_n_models=3
                    # param_grid={"a":[0,"aaa",2,3],
                    #             "b":["a0a","bbb",5,6]}
                                        
                    # Select (max_n_models-1) combinations of parameters AND add default (being the first parameter)
                    self.param_sampled = list(ParameterSampler(param_distributions=param_grid,
                                                          n_iter=max_n_models-1))

                    self.param_default = {k: param_grid.get(k,None)[0] for k in param_grid.keys()}                    
                    
                    # Combine lists of dicts
                    param_grid = [self.param_default]+self.param_sampled
                    
                    # Listify all values in dicts
                    param_grid = [{k:[v] for k,v in l.items()} for l in param_grid]
                    
                    # Convert to dict of lists
                    # param_grid = pd.DataFrame(param_grid).to_dict('list')
                    
                # Grid search over all parameters provided (sampled or not)    
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

    def cross_fit(self,X,y,shuffle=False):
        
        # Check X and Y
        X, y = check_X_Y(X, y)

        # Check n_cf_folds
        if self.n_cf_folds is None:
            raise Exception("""
                            When calling 'cross_fit()', one has to intentionally provide 'n_cf_folds' in the __init__
                            That is, 'n_cf_folds' is required to be an integer and not 'None'
                            """)     

        # Split into folds
        k_folds = KFold(n_splits=self.n_cf_folds, shuffle=shuffle)
        
        # Pre-allocate indices for the TEST set
        self.indices_ = np.zeros_like(a=y)
        self.y_pred_cf_ = np.zeros_like(a=y)
        
        for i,(train_index, test_index) in enumerate(k_folds.split(X)):
            
            # Increase cnt
            i += 1
            
            print(f"Cross-fitting fold {i}/{self.n_cf_folds}")
            
            # Fill indices
            self.indices_[test_index] = i

            # Copy estimator
            estimator_cf = deepcopy(self.estimator_cv)

            # Fit model using train indices
            estimator_cf.fit(X=X[train_index],
                             y=y[train_index])
        
            # Predict using test indices
            self.y_pred_cf_[test_index] = estimator_cf.predict(X=X[test_index])
    
            # Store estimator as attr
            setattr(self,f"estimator_{i}",estimator_cf)
    
        # Compute residuals
        self.y_res_cf_ = y - self.y_pred_cf_
    
        return self
    
    
        