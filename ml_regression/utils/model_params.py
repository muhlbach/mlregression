#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
from sklearn import (dummy,ensemble,gaussian_process,isotonic,kernel_ridge,
                     linear_model,neighbors,neural_network,svm,tree,)
from itertools import compress

# User
from .sanity_check import check_estimator

#------------------------------------------------------------------------------
# Model tools
#------------------------------------------------------------------------------
def compose_model(name, perform_estimator_check=True, verbose=True):
        
        # Modules from scikit-learn used for regression
        SCIKIT_REGRESSION_MODULES = ["dummy","ensemble","gaussian_process",
                                     "isotonic","kernel_ridge","linear_model",
                                     "neighbors","neural_network","svm","tree",]

        # Check if 'name' is an attribute of any Scikit-learn module
        scikit_regression_modules_check = [hasattr(eval(lib), name) for lib in SCIKIT_REGRESSION_MODULES]
        is_scikit = any(scikit_regression_modules_check)
        
        # Compose model
        if is_scikit:
            # Locate library
            scikit_module = list(compress(SCIKIT_REGRESSION_MODULES, scikit_regression_modules_check))[0]
            if verbose:
                print(f"Algorithm '{name}' is part of Scikit-learn, namely an attribute of: '{scikit_module}'")
                            
            # Instantiate model
            estimator = eval(scikit_module+"."+name+"()")
            
        else:
            raise NotImplementedError()

        # Model checks
        if perform_estimator_check:
            check_estimator(estimator=estimator)

        return estimator

#------------------------------------------------------------------------------
# Parameter tools
#------------------------------------------------------------------------------
def update_params(old_param, new_param, errors="raise"):
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


def get_param_grid_from_estimator(estimator):
    """
    Get the corresponding parameter grid from the estimator.
    The default parameter is *ALWAYS* the FIRST parameter listed, *UNLESS* the parameter is followed by a comment, e.g., "Default 100"
    Read the scikit-learn reference here: https://scikit-learn.org/stable/modules/classes.html# 
    """
    #--------------------------------------------------------------------------
    # from sklearn.dummy
    #--------------------------------------------------------------------------
    if isinstance(estimator, dummy.DummyRegressor):
        param_grid = {
            "strategy":["mean","median","quantile","constant"],
            "constant":None,
            "quantile":None
            }

    #--------------------------------------------------------------------------
    # from sklearn.ensemble
    #--------------------------------------------------------------------------
    # elif isinstance(estimator, ensemble.AdaBoostRegressor):
    #     raise NotImplementedError()
        
    # elif isinstance(estimator, ensemble.BaggingRegressor):
    #     raise NotImplementedError()
        
    # elif isinstance(estimator, ensemble.ExtraTreesRegressor):
    #     raise NotImplementedError()        

    # elif isinstance(estimator, ensemble.GradientBoostingRegressor):
    #     raise NotImplementedError()

    elif isinstance(estimator, ensemble.RandomForestRegressor):
        param_grid = {
            "n_estimators":500,                                                # Default 100 
            "criterion":'squared_error',
            "max_depth":[None,4,8,16],
            "min_samples_split":[2,4,8,16],
            "min_samples_leaf":[1,2,4,8],
            "min_weight_fraction_leaf":0.0,
            "max_features":['sqrt',1/4,1/3,1/2,2/3,'log2','auto'],             # Default 100
            "max_leaf_nodes":None,
            "min_impurity_decrease":0.0,
            "bootstrap":True,
            "oob_score":False,
            "n_jobs":None,
            "random_state":None,
            "verbose":0,
            "warm_start":False,
            "ccp_alpha":0.0,
            "max_samples":None
            }

    elif isinstance(estimator, ensemble.HistGradientBoostingRegressor):
        raise NotImplementedError()    

    #--------------------------------------------------------------------------
    # from sklearn.gaussian_process
    #--------------------------------------------------------------------------
    # elif isinstance(estimator, gaussian_process.GaussianProcessRegressor):
    #     raise NotImplementedError()    

    #--------------------------------------------------------------------------
    # from sklearn.isotonic
    #--------------------------------------------------------------------------
    # elif isinstance(estimator, isotonic.IsotonicRegression):
    #     raise NotImplementedError()    
    
    #--------------------------------------------------------------------------
    # from sklearn.kernel_ridge
    #--------------------------------------------------------------------------
    # elif isinstance(estimator, kernel_ridge.KernelRidge):
    #     raise NotImplementedError()    
    
    #--------------------------------------------------------------------------
    # from sklearn.linear_model
    #--------------------------------------------------------------------------
    elif isinstance(estimator, linear_model.LinearRegression):
        param_grid = {
            "fit_intercept":True,
            "normalize":'deprecated',
            "copy_X":True,
            "n_jobs":None,
            "positive":False
            }
        
    # elif isinstance(estimator, linear_model.Ridge):
    #     raise NotImplementedError()            

    # elif isinstance(estimator, linear_model.SGDRegressor):
    #     raise NotImplementedError()    
        
    # elif isinstance(estimator, linear_model.ElasticNet):
    #     raise NotImplementedError()    

    # elif isinstance(estimator, linear_model.Lars):
    #     raise NotImplementedError()    

    # elif isinstance(estimator, linear_model.Lasso):
    #     raise NotImplementedError()    

    # elif isinstance(estimator, linear_model.LassoLars):
    #     raise NotImplementedError()    

    # elif isinstance(estimator, linear_model.OrthogonalMatchingPursuit):
    #     raise NotImplementedError()            
        
    # elif isinstance(estimator, linear_model.ARDRegression):
    #     raise NotImplementedError()           

    # elif isinstance(estimator, linear_model.BayesianRidge):
    #     raise NotImplementedError()           

    # elif isinstance(estimator, linear_model.HuberRegressor):
    #     raise NotImplementedError()           

    # elif isinstance(estimator, linear_model.QuantileRegressor):
    #     raise NotImplementedError()           

    # elif isinstance(estimator, linear_model.RANSACRegressor):
    #     raise NotImplementedError()           

    # elif isinstance(estimator, linear_model.TheilSenRegressor):
    #     raise NotImplementedError()                   
        
    #--------------------------------------------------------------------------
    # from sklearn.neighbors
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    # from sklearn.neural_network
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    # from sklearn.svm
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    # from sklearn.tree
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Other
    #--------------------------------------------------------------------------
    else:
        try:
            param_grid = estimator.get_params()
        except:        
            raise NotImplementedError(f"Algorithm '{estimator}' is currently not implemented and has no 'get_params()'-method")
            
    return param_grid

  

    # # Lasso
    # lambda_max = (X.T @ y.values).abs().max()/X.shape[0]
     
    #     "Lasso" : {
    #         "model_params" : {
    #             "fit_intercept":True,
    #             "normalize":False,
    #             "precompute":False,
    #             "copy_X":True,
    #             "max_iter":100000,
    #             "tol":0.0001,
    #             "warm_start":False,
    #             "positive":False,
    #             "random_state":None,
    #             "selection":'cyclic'
    #             },
    #         "tuning_params" : {
    #             "alpha": np.exp(np.linspace(start=np.log(lambda_max), stop=np.log(lambda_max*0.000001), num=1000))
    #             }
    #     },
    #     "Ridge" : {
    #         "model_params" : {
    #             "fit_intercept":True,
    #             "normalize":False,
    #             "copy_X":True,
    #             "max_iter":None,
    #             "tol":0.0001,
    #             "solver":'auto',
    #             "random_state":None,
    #             },
    #         "tuning_params" : {
    #             "alpha": [0.001, 0.1, 1, 10, 100, 1000, 10000]
    #             }
    #     },
    #     "ElasticNet" : {
    #         "model_params" : {
    #             "fit_intercept":True,
    #             "normalize":False,
    #             "precompute":False,
    #             "max_iter":100000,
    #             "copy_X":True,
    #             "tol":0.0001,
    #             "warm_start":False,
    #             "positive":False,
    #             "random_state":None,
    #             "selection":'cyclic'
    #             },
    #         "tuning_params" : {
    #             "alpha": np.exp(np.linspace(start=np.log(lambda_max), stop=np.log(lambda_max*0.000001), num=1000)),
    #             "l1_ratio": [1/4, 1/2, 3/4, 1]
    #             }
    #     },
    #     "XGBRegressor" : {
    #         "model_params" : {
    #             'booster' : 'gbtree',
    #             'objective' : "reg:squarederror",
    #             'verbosity' : 1,
    #             'tree_method' : 'auto',
    #             'validate_parameters' : 1
    #             },
    #         "tuning_params" : {
    #             'learning_rate': [0.3, 0.1, 0.01, 0.5],
    #             'gamma': [0, 0.5],
    #             'max_depth': [6, 8, 4, 2],
    #             'min_child_weight': [1, 4, 8],
    #             'subsample': [1, 0.7],
    #             'colsample_bytree': [1, 2/3, 1/3],
    #             'n_estimators': [100, 200, 400]
    #             }
    #     },
    #     "RandomForestRegressor" : {
    #         "model_params" : {
    #             "criterion":'mse',
    #             "min_weight_fraction_leaf":0.0,
    #             'min_impurity_decrease':0.0,
    #             'min_impurity_split':None,
    #             'bootstrap': True,
    #             'oob_score':False,
    #             'n_jobs':None,
    #             'random_state':None,
    #             'verbose':0,
    #             'warm_start':False,
    #             'ccp_alpha':0.0,
    #             'max_samples':None
    #             },
    #         "tuning_params" : {
    #             'n_estimators': 500,
    #             'min_samples_split': [2,4,8,16],
    #             'min_samples_leaf': [1,2,4,8],
    #             'max_features': [1/4,1/3,1/2,2/3, 'sqrt','log2'],
    #             'max_leaf_nodes': None,
    #             'max_depth': [4,8,16,None]
    #             }
    #     },
    #     "ExtraTreesRegressor" : {
    #         "model_params" : {
    #             "criterion":'mse',
    #             "min_weight_fraction_leaf":0.0,
    #             'min_impurity_decrease':0.0,
    #             'min_impurity_split':None,
    #             'bootstrap': True,
    #             'oob_score':False,
    #             'n_jobs':None,
    #             'random_state':None,
    #             'verbose':0,
    #             'warm_start':False,
    #             'ccp_alpha':0.0,
    #             'max_samples':None
    #             },
    #         "tuning_params" : {
    #             'n_estimators': 500,
    #             'min_samples_split': [2,4,8,16],
    #             'min_samples_leaf': [1,2,4,8],
    #             'max_features': [1/4,1/3,1/2,2/3, 'sqrt','log2'],
    #             'max_leaf_nodes': None,
    #             'max_depth': [4,8,16,None]
    #             }
    #     },
    #     "SGDRegressor" : {
    #         "model_params" : {
    #             "loss":'squared_loss',
    #             "penalty":'l2',
    #             "l1_ratio":0.15,
    #             "fit_intercept":True,
    #             "max_iter":10000,
    #             "tol":0.0001,
    #             "shuffle":True,
    #             "verbose":0,
    #             "epsilon":0.1,
    #             "random_state":None,
    #             "eta0":0.01,
    #             "power_t":0.25,
    #             "early_stopping":True,
    #             "validation_fraction":0.1,
    #             "n_iter_no_change":50,
    #             "warm_start":False, 
    #             "average":False
    #             },
    #         "tuning_params" : {
    #             "alpha":[0.0001, 0.001, 0.01, 0.00001, 0.000001],
    #             "learning_rate":['invscaling', 'constant', 'optimal', 'adaptive'],
    #             }
    #     },
    #     "ARDRegression" : {
    #         "model_params" : {
    #             "n_iter":1000,
    #             "tol":0.0001,
    #             "compute_score":False,
    #             "threshold_lambda":10000.0,
    #             "fit_intercept":True,
    #             "normalize":False,
    #             "copy_X":True,
    #             "verbose":False
    #             },
    #         "tuning_params" : {
    #             "alpha_1":[1e-06, 1e-07, 1e-08, 1e-05, 1e-04],
    #             "alpha_2":[1e-06, 1e-07, 1e-08, 1e-05, 1e-04],
    #             "lambda_1":[1e-06, 1e-07, 1e-08, 1e-05, 1e-04],
    #             "lambda_2":[1e-06, 1e-07, 1e-08, 1e-05, 1e-04],
    #             }
    #     },
    #     "BayesianRidge" : {
    #         "model_params" : {
    #             "n_iter":1000,
    #             "tol":0.0001,
    #             'alpha_init':None,
    #             'lambda_init':None,
    #             "compute_score":False,
    #             "fit_intercept":True,
    #             "normalize":False,
    #             "copy_X":True,
    #             "verbose":False
    #             },
    #         "tuning_params" : {
    #             "alpha_1":[1e-06, 1e-07, 1e-08, 1e-05, 1e-04],
    #             "alpha_2":[1e-06, 1e-07, 1e-08, 1e-05, 1e-04],
    #             "lambda_1":[1e-06, 1e-07, 1e-08, 1e-05, 1e-04],
    #             "lambda_2":[1e-06, 1e-07, 1e-08, 1e-05, 1e-04],
    #             }
    #     },
    #     "HuberRegressor" : {
    #         "model_params" : {
    #             "max_iter":10000,
    #             "warm_start":False,
    #             "fit_intercept":True,
    #             "tol":0.00001
    #             },
    #         "tuning_params" : {
    #             "epsilon":[1.35, 1.1,1.2,1.3,1.4,1.5,175],
    #             "alpha":[0.0001, 0.001, 0.01, 0.00001, 0.000001],                                        
    #             }
    #     },
    #     "RANSACRegressor" : {
    #         "model_params" : {
    #             "min_samples":None,
    #             "residual_threshold":None, 
    #             "is_data_valid":None,
    #             "is_model_valid":None,
    #             "max_trials":100,
    #             "max_skips":np.inf,
    #             "stop_n_inliers":np.inf,
    #             "stop_score":np.inf,
    #             "stop_probability":0.99,
    #             "loss":'absolute_loss',
    #             "random_state":None
    #             },
    #         "tuning_params" : {
    #             "base_estimator":linear_model.LinearRegression(),
    #             }
    #     },
    #     "TheilSenRegressor" : {
    #         "model_params" : {
    #             "fit_intercept":True,
    #             "copy_X":True,
    #             "max_subpopulation":10000,
    #             "n_subsamples":None,
    #             "max_iter":10000,
    #             "tol":0.0001,
    #             "random_state":None,
    #             "n_jobs":None,
    #             "verbose":False
    #             },
    #         "tuning_params" : {}
    #     },
    #     "PassiveAggressiveRegressor" : {
    #         "model_params" : {
    #             "fit_intercept":True,
    #             "max_iter":10000,
    #             "tol":0.0001,
    #             "early_stopping":True,
    #             "validation_fraction":0.1,
    #             "n_iter_no_change":50,
    #             "shuffle":True,
    #             "verbose":0,
    #             "loss":'epsilon_insensitive',
    #             "epsilon":0.1,
    #             "random_state":None,
    #             "warm_start":False,
    #             "average":False
    #             },
    #         "tuning_params" : {
    #             "C":[0.1,1.0,10],
    #             }
    #     },
    #     "KNeighborsRegressor" : {
    #         "model_params" : {
    #             "algorithm":'auto',
    #             "leaf_size":30,
    #             "p":2,
    #             "metric":'minkowski',
    #             "metric_params":None,
    #             "n_jobs":None
    #             },
    #         "tuning_params" : {
    #             "n_neighbors":[5,1,2,4,8,16,32],
    #             "weights":['distance','uniform'],
    #             }
    #     },
    #     "MLPRegressor" : {
    #         "model_params" : {
    #             "solver":'adam',
    #             "batch_size":'auto',
    #             "learning_rate_init":0.001,
    #             "power_t":0.5,
    #             "max_iter":10000,
    #             "shuffle":True,
    #             "random_state":None,
    #             "tol":0.0001,
    #             "verbose":False,
    #             "warm_start":False,
    #             "momentum":0.9,
    #             "nesterovs_momentum":True,
    #             "early_stopping":True,
    #             "validation_fraction":0.1,
    #             "beta_1":0.9,
    #             "beta_2":0.999,
    #             "epsilon":1e-08,
    #             "n_iter_no_change":50,
    #             "max_fun":15000
    #             },
    #         "tuning_params" : {
    #             "hidden_layer_sizes":[(2,), (2,4), (2,4,8,), (2,4,8,16,), (2,4,8,16,32),
    #                                   (4,), (4,8), (4,8,16), (4,8,16,32),
    #                                   (8,), (8,16), (8,16,32),
    #                                   (16,), (16,32),
    #                                   (32,)],
    #             "activation":['relu'],
    #             "alpha":[1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06],
    #             "learning_rate":['constant', 'adaptive'],
    #             }
    #     },
    #     "LinearSVR" : {
    #         "model_params" : {
    #             "epsilon":0.0,
    #             "tol":0.0001,
    #             "loss":'epsilon_insensitive',
    #             "fit_intercept":True,
    #             "intercept_scaling":1.0,
    #             "dual":dual_par,
    #             "verbose":0,
    #             "random_state":None,
    #             "max_iter":1000000
    #             },
    #         "tuning_params" : {
    #             "C":[0.1,1,2,5,10],
    #             }
    #     },
    #     "NuSVR" : {
    #         "model_params" : {
    #             "degree":3,
    #             "gamma":'scale',
    #             "coef0":0.0,
    #             "shrinking":True,
    #             "tol":0.001,
    #             "cache_size":200,
    #             "verbose":False,
    #             "max_iter":- 1
    #             },
    #         "tuning_params" : {
    #             "C":[0.1,1,2,5,10],
    #             "nu":[1/4,1/2, 3/4],
    #             "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
    #             }
    #     }                          
    # }
