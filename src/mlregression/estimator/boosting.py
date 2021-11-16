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

# Lazy implementation:
# class XGBRegressor(xgboost.XGBRegressor):    
#     def __init__(self, **kwargs):  
#         super().__init__(**kwargs)


#------------------------------------------------------------------------------
# LightGBM
#------------------------------------------------------------------------------
