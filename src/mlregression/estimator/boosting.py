#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import xgboost as xgb

#------------------------------------------------------------------------------
# XGBoost
#------------------------------------------------------------------------------
class XGBRegressor(xgb.XGBRegressor):
    """
    This class copies verbatim the XGBoost regressor
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
                  n_jobs=None,
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
                  ):
        # self.n_estimators=n_estimators                           
        # self.max_depth=max_depth
        # self.learning_rate=learning_rate
        # self.verbosity=verbosity
        # self.booster=booster
        # self.tree_method=tree_method
        # self.n_jobs=n_jobs
        # self.gamma=gamma
        # self.min_child_weight=min_child_weight
        # self.max_delta_step=max_delta_step
        # self.subsample=subsample
        # self.colsample_bytree=colsample_bytree
        # self.colsample_bylevel=colsample_bylevel
        # self.colsample_bynode=colsample_bynode
        # self.reg_alpha=reg_alpha
        # self.reg_lambda=reg_lambda
        # self.scale_pos_weight=scale_pos_weight
        # self.base_score=base_score
        # self.random_state=random_state
        # self.missing=missing
        # self.num_parallel_tree=num_parallel_tree
        # self.monotone_constraints=monotone_constraints
        # self.interaction_constraints=interaction_constraints
        # self.importance_type=importance_type
        # self.gpu_id=gpu_id
        # self.validate_parameters=validate_parameters
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
            validate_parameters=validate_parameters
            )

# # Lazy implementation:
# class XGBRegressor(xgb.XGBRegressor):    
#     def __init__(self, **kwargs):  
#         super().__init__(**kwargs)


#------------------------------------------------------------------------------
# LightGBM
#------------------------------------------------------------------------------
