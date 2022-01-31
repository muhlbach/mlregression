#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
from sklearn import (dummy,ensemble,gaussian_process,isotonic,kernel_ridge,
                     linear_model,neighbors,neural_network,svm,tree,)
from itertools import compress

# User
from ..estimator import (boosting)
from .sanity_check import check_estimator
from .exceptions import WrongInputException

#------------------------------------------------------------------------------
# Model tools
#------------------------------------------------------------------------------
def compose_model(estimator_name, perform_estimator_check=True, verbose=True):
    """
    Instantitate model from string. Note that this uses all params from .get_params() in case we overwrite models (e.g. boosting)
    """
    # Modules from scikit-learn used for regression
    SCIKIT_REGRESSION_MODULES = ["dummy","ensemble","gaussian_process",
                                 "isotonic","kernel_ridge","linear_model",
                                 "neighbors","neural_network","svm","tree",]
    
    # Internal modules used for regression
    NONSCIKIT_REGRESSION_MODULES = ["boosting",]
    
    REGRESSION_MODULES = [SCIKIT_REGRESSION_MODULES,NONSCIKIT_REGRESSION_MODULES]
    
    # Count the number of modules in which we find the model name
    n_modules_found_in = 0
    
    for reg_module in REGRESSION_MODULES:
        
        # Check if 'estimator_name' is an attribute of any module
        module_check = [hasattr(eval(lib), estimator_name) for lib in reg_module]
    
        is_found = any(module_check)

        # Compose model            
        if is_found:
            
            # Check counter
            if n_modules_found_in>1:
                raise Exception(f"""Algorithm '{estimator_name}' is part of multiple modules, namely at least two of
                                \n{SCIKIT_REGRESSION_MODULES}
                                \n{NONSCIKIT_REGRESSION_MODULES}
                                """)
            
            # Locate module
            module = list(compress(reg_module, module_check))[0]

            # Instantiate model
            estimator = eval(module+"."+estimator_name+"()")

            # Increase model counter
            n_modules_found_in += 1

    if n_modules_found_in==0:
        raise Exception(f"""Algorithm '{estimator_name}' is NOT part any module, namely neither of
                        \n{SCIKIT_REGRESSION_MODULES}
                        \n{NONSCIKIT_REGRESSION_MODULES}
                        """)

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

def isinstance_double(obj, class_or_tuple, how="all"):
    ALLOWED_HOW = ["all", "any"]
    
    if how=="all":
        is_instance = isinstance(obj,class_or_tuple) and (obj.__class__.__name__==class_or_tuple.__name__)
    elif how=="any":
        is_instance = isinstance(obj,class_or_tuple) or (obj.__class__.__name__==class_or_tuple.__name__)    
    else:
        raise WrongInputException(input_name="how",
                                  provided_input=how,
                                  allowed_inputs=ALLOWED_HOW)
    
    return is_instance


def get_param_grid_from_estimator(estimator):
    """
    Get the corresponding parameter grid from the estimator.
    The default parameter is *ALWAYS* the FIRST parameter listed, *UNLESS* the parameter is followed by a comment, e.g., "Default 100"
    Read the scikit-learn reference here: https://scikit-learn.org/stable/modules/classes.html# 
    """
    #--------------------------------------------------------------------------
    # from ..estimator.boosting
    #--------------------------------------------------------------------------        
    if isinstance_double(estimator, boosting.XGBRegressor):
        param_grid = {
            "n_estimators":[100,200,500],
            "max_depth":[None,2,4,8,16],
            "learning_rate":[1, 0.8, 0.5, 0.3, 0.1, 0.01],
            "verbosity":0,
            # "objective":'reg:squarederror', # Leads to RuntimeError: Cannot clone object XGBRegressor(...), as the constructor either does not set or modifies parameter objective
            "booster":None,
            "tree_method":None,
            "gamma":[None, 0, 0.5],
            "min_child_weight":[None,1,2,4,8],
            "max_delta_step":None,
            "subsample":[0.8,1,0.5],
            "colsample_bytree":[None,1,2/3,1/3],
            "colsample_bylevel":None,
            "colsample_bynode":[0.8,1,0.5],
            "reg_alpha":None,
            "reg_lambda":[1e-05,0],
            "scale_pos_weight":None,
            "base_score":None,
            "num_parallel_tree":None,
            "monotone_constraints":None,
            "interaction_constraints":None,
            "importance_type":'gain',
            }
    
    elif isinstance_double(estimator, boosting.LGBMegressor):
        param_grid = {
            "boosting_type":'gbdt',
            "num_leaves":[31,11,21,41],
            "max_depth":[-1,2,4,8,16],
            "learning_rate":[0.1,0.8, 0.5, 0.3, 0.01],
            "n_estimators":[100,200,500],
            "subsample_for_bin":200000,
            "objective":'regression',
            "class_weight":None,
            "min_split_gain":0.0,
            "min_child_weight":0.001,
            "min_child_samples":[20,4,8,16],
            "subsample":[1.0,0.8,0.5],
            "subsample_freq":0,
            "colsample_bytree":[1.0,2/3,1/3],
            "reg_alpha":[0.0,1e-05],
            "reg_lambda":[0.0,1e-05],
            "importance_type":'split'
            }  
    
    #--------------------------------------------------------------------------
    # from sklearn.dummy
    #--------------------------------------------------------------------------
    elif isinstance_double(estimator, dummy.DummyRegressor):
        param_grid = {
            "strategy":["mean","median"],
            }

    #--------------------------------------------------------------------------
    # from sklearn.ensemble
    #--------------------------------------------------------------------------
    # elif isinstance_double(estimator, ensemble.AdaBoostRegressor):
    #     raise NotImplementedError()
        
    # elif isinstance_double(estimator, ensemble.BaggingRegressor):
    #     raise NotImplementedError()
        
    elif isinstance_double(estimator, ensemble.ExtraTreesRegressor):
        param_grid = {
            "n_estimators":200,                                                # Default 100
            'criterion':'squared_error',
            'max_depth':[None,2,4,8,16],
            'min_samples_split':[2,4,8,16],
            'min_samples_leaf':[1,2,4,8],
            'min_weight_fraction_leaf':0.0,
            'max_features':['sqrt',1/4,1/3,1/2,2/3,'log2','auto'],             # Default 'auto'
            'max_leaf_nodes':None,
            'min_impurity_decrease':0.0,
            'bootstrap':False,
            'oob_score':False,
            'ccp_alpha':0.0,
            'max_samples':None
            }   

    elif isinstance_double(estimator, ensemble.GradientBoostingRegressor):
        param_grid = {
            "loss":'squared_error',
            'learning_rate':[0.1,0.8, 0.5, 0.3, 0.01],
            'n_estimators':[100,200,500],
            'subsample':[1.0,0.8,0.5],
            'criterion':'friedman_mse',
            'min_samples_split':[2,4,8,16],
            'min_samples_leaf':[1,2,4,8],
            'min_weight_fraction_leaf':0.0,
            'max_depth':[3,2,4,8,16],
            'min_impurity_decrease':0.0,
            'max_features':['sqrt',1/4,1/3,1/2,2/3,'log2',None],               # Default None
            'alpha':0.9,
            'verbose':0,
            'max_leaf_nodes':None,
            'validation_fraction':0.1,
            'n_iter_no_change':None,
            'tol':0.0001,
            'ccp_alpha':0.0
            }
        
    elif isinstance_double(estimator, ensemble.RandomForestRegressor):
        param_grid = {
            "n_estimators":200,                                                # Default 100
            "criterion":'squared_error',
            "max_depth":[None,2,4,8,16],
            "min_samples_split":[2,4,8,16],
            "min_samples_leaf":[1,2,4,8],
            "min_weight_fraction_leaf":0.0,
            "max_features":['sqrt',1/4,1/3,1/2,2/3,'log2','auto'],             # Default 'auto'    
            "max_leaf_nodes":None,
            "min_impurity_decrease":0.0,
            "bootstrap":True,
            "oob_score":False,
            "n_jobs":None,
            "random_state":None,
            "verbose":0,
            "warm_start":False,
            # "ccp_alpha":0.0, # Leads to RuntimeError: Cannot clone object RandomForestRegressor(...), as the constructor either does not set or modifies parameter ccp_alpha
            "max_samples":None
            }

    # elif isinstance_double(estimator, ensemble.HistGradientBoostingRegressor):
    #     raise NotImplementedError()    

    #--------------------------------------------------------------------------
    # from sklearn.gaussian_process
    #--------------------------------------------------------------------------
    # elif isinstance_double(estimator, gaussian_process.GaussianProcessRegressor):
    #     raise NotImplementedError()    

    #--------------------------------------------------------------------------
    # from sklearn.isotonic
    #--------------------------------------------------------------------------
    # elif isinstance_double(estimator, isotonic.IsotonicRegression):
    #     raise NotImplementedError()    
    
    #--------------------------------------------------------------------------
    # from sklearn.kernel_ridge
    #--------------------------------------------------------------------------
    # elif isinstance_double(estimator, kernel_ridge.KernelRidge):
    #     raise NotImplementedError()    
    
    #--------------------------------------------------------------------------
    # from sklearn.linear_model
    #--------------------------------------------------------------------------
    elif isinstance_double(estimator, linear_model.LinearRegression):
        param_grid = {
            "fit_intercept":True,
            "copy_X":True,
            "n_jobs":None,
            "positive":False
            }
                
    elif isinstance_double(estimator, linear_model.Ridge):
        param_grid = {
            'alpha':[1.0,0.001, 0.1, 10, 100, 1000, 10000], 
            'fit_intercept':True,
            'copy_X':True,
            'max_iter':None,
            'tol':0.001,
            'solver':'auto',
            'random_state':None
            }
        

    elif isinstance_double(estimator, linear_model.RidgeCV):
        param_grid = {
            'alphas':(1.0,0.001, 0.1, 10, 100, 1000, 10000),
            'fit_intercept':True,
            'scoring':None,
            'gcv_mode':None,
            'store_cv_values':False,
            'alpha_per_target':False
            }
        
    # elif isinstance_double(estimator, linear_model.SGDRegressor):
    #     raise NotImplementedError()    
        
    elif isinstance_double(estimator, linear_model.ElasticNet):
        param_grid = {
            'alpha':np.exp(np.linspace(start=np.log(100), stop=np.log(100*0.000001), num=1000)),
            'l1_ratio':[0.5,1/4,3/4,1],
            'fit_intercept':True,
            'precompute':False,
            'max_iter':10,
            'tol':0.0001,
            'warm_start':False,
            'positive':False,
            'selection':'cyclic'
            }

    elif isinstance_double(estimator, linear_model.ElasticNetCV):
        param_grid = {
            'l1_ratio':[0.5,1/4,3/4,1],
            'eps':0.001,
            'n_alphas':1000,
            'fit_intercept':True,
            'precompute':'auto',
            'max_iter':1000,
            'tol':0.0001,
            'copy_X':True,
            'positive':False,
            'selection':'cyclic'
            }

    elif isinstance_double(estimator, linear_model.Lars):        
        param_grid = {
            'fit_intercept':True,
            'precompute':'auto',
            'n_nonzero_coefs':[500,100,200,300,400,np.inf],
            'eps':2.220446049250313e-16,
            'copy_X':True,
            'fit_path':True,
            'jitter':None,
            }

    elif isinstance_double(estimator, linear_model.LarsCV):
        param_grid = {
            'fit_intercept':True,
            'max_iter':500,
            'precompute':'auto',
            'max_n_alphas':1000,
            'eps':2.220446049250313e-16,
            'copy_X':True
            }
        
    elif isinstance_double(estimator, linear_model.Lasso):
        param_grid = {
            'alpha':np.exp(np.linspace(start=np.log(100), stop=np.log(100*0.000001), num=1000)),
            'fit_intercept':True,
            'precompute':False,
            'copy_X':True,
            'max_iter':1000,
            'tol':0.0001,
            'warm_start':False,
            'positive':False,
            'selection':'cyclic'
            }

    elif isinstance_double(estimator, linear_model.LassoCV):
        param_grid = {
            'eps':0.001,
            'n_alphas':1000,
            'fit_intercept':True,
            'precompute':'auto',
            'max_iter':1000,
            'tol':0.0001,
            'copy_X':True,
            'positive':False,
            'selection':'cyclic'
            }

    elif isinstance_double(estimator, linear_model.LassoLars):
        param_grid = {
            'alpha':np.exp(np.linspace(start=np.log(100), stop=np.log(100*0.000001), num=1000)),
            'fit_intercept':True,
            'precompute':'auto',
            'max_iter':500,
            'eps':2.220446049250313e-16,
            'copy_X':True,
            'fit_path':True,
            'positive':False,
            'jitter':None,
            }

    elif isinstance_double(estimator, linear_model.LassoLarsCV):
        param_grid = {
            'fit_intercept':True,
            'max_iter':500,
            'precompute':'auto',
            'max_n_alphas':1000,
            'eps':2.220446049250313e-16,
            'copy_X':True,
            'positive':False
            }
        
    # elif isinstance_double(estimator, linear_model.OrthogonalMatchingPursuit):
    #     raise NotImplementedError()            
        
    # elif isinstance_double(estimator, linear_model.ARDRegression):
    #     raise NotImplementedError()           

    # elif isinstance_double(estimator, linear_model.BayesianRidge):
    #     raise NotImplementedError()           

    # elif isinstance_double(estimator, linear_model.HuberRegressor):
    #     raise NotImplementedError()           

    # elif isinstance_double(estimator, linear_model.QuantileRegressor):
    #     raise NotImplementedError()           

    # elif isinstance_double(estimator, linear_model.RANSACRegressor):
    #     raise NotImplementedError()           

    # elif isinstance_double(estimator, linear_model.TheilSenRegressor):
    #     raise NotImplementedError()                   
        
    #--------------------------------------------------------------------------
    # from sklearn.neighbors
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    # from sklearn.neural_network
    #--------------------------------------------------------------------------
    elif isinstance_double(estimator, neural_network.MLPRegressor):
        param_grid = {
            'hidden_layer_sizes':[(100,), (2,), (2,4), (2,4,8,), (2,4,8,16,), (2,4,8,16,32),
                                          (4,), (4,8), (4,8,16), (4,8,16,32),
                                          (8,), (8,16), (8,16,32),
                                          (16,), (16,32),
                                          (32,)],
            'activation':'relu',
            'solver':'adam',
            'alpha':[0.0001,1.e+01, 1.e+02, 1.e+03, 1.e+05, 1.e+06],
            'batch_size':'auto',
            'learning_rate':['constant', 'invscaling', 'adaptive'],
            'learning_rate_init':0.001,
            'power_t':0.5,
            'max_iter':200,
            'shuffle':True,
            'tol':0.0001,
            'momentum':0.9,
            'nesterovs_momentum':True,
            'early_stopping':True,
            'validation_fraction':0.1,
            'beta_1':0.9,
            'beta_2':0.999,
            'epsilon':1e-08,
            'n_iter_no_change':10,
            'max_fun':15000
            }
        
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
    #
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
