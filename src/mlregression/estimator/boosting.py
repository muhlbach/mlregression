#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import xgboost as xgb
import lightgbm as lgbm

a=xgb.XGBRegressor()



a.__init__()

#------------------------------------------------------------------------------
# XGBoost
#------------------------------------------------------------------------------
class XGBRegressor(xgb.XGBRegressor):
    """
    This class copies verbatim the XGBoost regressor as of version 1.5.0
    See: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                  n_estimators=200,                                             # Default 100 
                  max_depth=None,
                  learning_rate=1,
                  verbosity=0,
                  objective='reg:squarederror',
                  booster=None,
                  tree_method=None,
                  n_jobs=1,
                  gamma=None,
                  min_child_weight=None,
                  max_delta_step=None,
                  subsample=0.8,
                  colsample_bytree=None,
                  colsample_bylevel=None,
                  colsample_bynode=0.8,
                  reg_alpha=None,
                  reg_lambda=1e-05,
                  scale_pos_weight=None,
                  base_score=None,
                  random_state=1991,
                  missing=np.nan,
                  num_parallel_tree=None,
                  monotone_constraints=None,
                  interaction_constraints=None,
                  importance_type='gain',
                  gpu_id=None,
                  validate_parameters=None,
                  enable_categorical=False,
                  predictor=None
                  ):
        super().__init__(
            n_estimators=n_estimators,                                             
            max_depth=max_depth,
            learning_rate=learning_rate,
            verbosity=verbosity,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
            missing=missing,
            num_parallel_tree=num_parallel_tree,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            importance_type=importance_type,
            gpu_id=gpu_id,
            validate_parameters=validate_parameters,
            enable_categorical=enable_categorical,
            predictor=predictor,
            )

# # Lazy implementation:
# class XGBRegressor(xgb.XGBRegressor):    
#     def __init__(self, **kwargs):  
#         super().__init__(**kwargs)


#------------------------------------------------------------------------------
# LightGBM
#------------------------------------------------------------------------------
class LGBMegressor(lgbm.LGBMRegressor):
    """
    This class copies verbatim the LightGBM regressor as of version 3.2.1
    See: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm-lgbmregressor
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 boosting_type='gbdt',
                 num_leaves=31,
                 max_depth=-1,
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample_for_bin=200000,
                 objective='regression',
                 class_weight=None,
                 min_split_gain=0.0,
                 min_child_weight=0.001,
                 min_child_samples=20,
                 subsample=1.0,
                 subsample_freq=0,
                 colsample_bytree=1.0,
                 reg_alpha=0.0,
                 reg_lambda=0.0,
                 random_state=None,
                 n_jobs=1,
                 silent='warn',
                 importance_type='split'
                 ):
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type
            )