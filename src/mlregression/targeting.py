#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import bodyguard as bg
import numpy as np
import pandas as pd
import re
from sklearn.utils import resample
import statsmodels.api as sm
from scipy.stats import t as student_t
# User
from .estimator.feature_selection import LassoLambda, LassoCV, LassoIC
from .utils.sanity_check import check_X_Y, check_X
from .estimator.linear_model import OLS
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class LassoTargeter():
    """
    This class implements Lasso as a selection method combining LassoLambda, LassoCV, LassoIC, all building on LassoBase
    All function calls go through self.estimator
    
    "method" = {"lassocv","elasticnetcv", "bic", "aic", "ic", "lambda"}
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 method="bic",
                 **kwargs
                 ):
        self.method=method
        
        bg.sanity_check.check_str(x=method,
                                  allowed=self.METHOD_ALLOWED,
                                  name="method")
        
        # Based on cross-validation
        if (self.method=="cv") | (self.method in self.CV_METHODS):
            self.estimator = LassoCV(method="lasso",
                                     **kwargs)
            
            if any(self.method==x for x in self.CV_METHODS):
                method_clean = re.sub(pattern="cv_", repl="", string=self.method)
                self.estimator.method = method_clean

        # Based on information-criterion
        elif (self.method=="ic") | (self.method in self.IC_METHODS):
            self.estimator = LassoIC(criterion="bic",
                                     **kwargs)
            
            if any(self.method==x for x in self.IC_METHODS):
                self.estimator.criterion = self.method
        
        # Based on lambda-tuning to specific number of targets
        elif (self.method=="lambda"):
            self.estimator = LassoLambda(**kwargs)
    
    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    CV_METHODS = ["cv_lasso", "cv_elasticnet"]
    IC_METHODS = ["bic", "aic"]
    OTHER_METHODS = ["lambda", "cv", "ic"]
    METHOD_ALLOWED = CV_METHODS+IC_METHODS+OTHER_METHODS
    
    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------
    def fit(self,X,y):
        self.estimator.fit(X=X,y=y)
        self.mask_features = self.estimator.mask_features
        
        return self
    
    def transform(self,X):
        return self.estimator.transform(X=X)

    def fit_transform(self,X,y):
        return self.estimator.fit_transform(X=X,y=y)
    

class PostLasso(object):
    """
    This class implements Post-Lasso and obtain confidence intervals by bootstrapping
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 n_bootstrap=100,
                 targeting_fraction=0.5,
                 targeting_method="bic",
                 significance_level=0.95,
                 fit_intercept=True,
                 inversion_method="pinv",
                 cov_type="HC1",
                 cov_kwds=None,
                 use_t=True,
                 **kwargs
                 ):
        self.n_bootstrap = n_bootstrap
        self.targeting_fraction = targeting_fraction
        self.targeting_method = targeting_method
        self.significance_level = significance_level
        self.fit_intercept = fit_intercept
        self.inversion_method = inversion_method
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.use_t = use_t
        
        # Initialize
        self.targeter = LassoTargeter(method=self.targeting_method,
                                      **kwargs)
        
        self.inferencer = OLS(fit_intercept=self.fit_intercept,
                              method=self.inversion_method,
                              cov_type=self.cov_type,
                              cov_kwds=self.cov_kwds,
                              use_t=self.use_t)
        
        # Translate significance levels to alphas
        self.alpha = [round((1-self.significance_level)/2,3)]
        
        self.alphas = self.alpha[::-1] + [1-a for a in self.alpha[::1]]
        
        
        self.lower_qnt = self.alpha[0]
        self.upper_qnt = 1-self.alpha[0]

    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------
    def fit(self,X,y):
        
        # Sanity check but NOT convert to nparray
        X, y = check_X_Y(X=X,Y=y,convert_to_nparray=False)
                
        if self.fit_intercept:
            X = sm.add_constant(data=X,prepend=True, has_constant='skip')
        
        # Full sample targeting
        mask_features_full_sample = self.targeter.fit(X=X,y=y).mask_features
        
        # Subset
        features_names = bg.lists.subset(l=X.columns.tolist(),
                                         boolean_l=mask_features_full_sample.tolist())
        
        # Convert
        features_names = [str(i) for i in features_names]
        
        # Pre-allocate
        coef = []
                
        for b in range(self.n_bootstrap):
            
            X_bootstrap, y_bootstrap = resample(X, y, 
                                                replace=True,
                                                n_samples=None,
                                                random_state=None,
                                                stratify=None)
            
            # Find unique indices (some are replicated due to bootstrap)
            unique_index = y_bootstrap.index.unique().tolist() 
            
            # Draw random set of unique indices
            targeting_index = np.random.choice(a=unique_index, size=int(self.targeting_fraction*len(unique_index)), replace=False)
            
            # Generate mask of indices (np.array of False/True)
            targeting_mask = pd.Series(False, index=y_bootstrap.index)
            targeting_mask.loc[targeting_index] = True
            targeting_mask = np.array(targeting_mask)
            
            # Subset data for targeting and inference
            X_targeting, y_targeting = X_bootstrap.loc[targeting_mask], y_bootstrap.loc[targeting_mask]
            X_inference, y_inference = X_bootstrap.loc[~targeting_mask], y_bootstrap.loc[~targeting_mask]
            
            # Obtain features from targeting data
            mask_features = self.targeter.fit(X=X_targeting,y=y_targeting).mask_features
            
            # Obtain coefficients from inference data
            coef_bootstrap = self.inferencer.fit(X=X_inference.iloc[:, mask_features],
                                                  y=y_inference).coef_

            # Store results
            coef.append(coef_bootstrap)


        # Concatenate all            
        df_coef = pd.concat(objs=coef,axis=1)

        # Subset to full-sample
        df_coef = df_coef.loc[features_names]
        
        # Median estimate
        df_coef_median = df_coef.median(axis=1)
        
        # Confidence interval (quantiles of empirical distribution)
        df_conf_int = df_coef.quantile(q=self.alphas, axis=1, numeric_only=False, interpolation='linear').transpose()
        
        dof = y.shape[0]-mask_features_full_sample.sum()
        
        # Back out standard errors
        df_se = (df_conf_int[self.upper_qnt] - df_coef_median) / student_t.ppf(q=self.upper_qnt, df=dof, loc=0, scale=1)

        # Implied t-stats
        df_t = df_coef_median.divide(df_se)
        
        # Implied p-values
        df_p = student_t.pdf(x=df_t, df=dof, loc=0, scale=1)

        self.results_ = pd.DataFrame(data={"coef":df_coef_median,
                                           "se":df_se,
                                           "t-stat":df_t,
                                           "p-value":df_p},
                                    index=df_coef.index)        
        
        # Save each
        for col in self.results_.columns:
            # Prettify string
            attr_name = bg.strings.make_string_compatible(string=col)
            
            if not attr_name.endswith("_"):
                attr_name = attr_name+"_"
            
            # Set attribute
            setattr(self,attr_name,self.results_[col])
            
        # Sanity check
        
        # ols = OLS()
        
        # ols.fit(X=X.iloc[:,mask_features_full_sample],
        #         y=y)
        # ols.summary()
        
        


    def target(self,X,y=None):
        return self.estimator.target(X=X,y=y)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    