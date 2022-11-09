#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import bodyguard as bg
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC, lasso_path
from sklearn.linear_model import LassoCV as ScikitLassoCV
from sklearn.linear_model import ElasticNetCV as ScikitElasticNetCV
from sklearn.linear_model import RidgeCV

from ..utils.sanity_check import check_X_Y, check_X
from .linear_model import OLS
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class LassoBase(object):
    """
    Implemented base methods for lasso-targeting
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------------    
    def _fit(self,X,y):
        """
        This calls the underlying self.estimator.fit()
        """
        
        # Fit
        self.estimator.fit(X=X,y=y)
        
        # Get coefficients
        coef_ = self.estimator.coef_
        
        # Get boolean of features to include
        self.mask_features = (coef_ != 0).reshape(-1)
        
        # Update fitted status
        self.is_fitted = True

        return self     

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------    
    def preprocess(self,X,y):
        
        # Breaks links
        X = X.copy()
        
        self.X_type_ = type(X)
        self.y_type_ = type(y)
        
        # Check dimensions
        X, y = check_X_Y(X=X, Y=y)
        
        # Instantiate
        scaler = StandardScaler(with_mean=True, with_std=True)
        
        # Scale
        X = scaler.fit_transform(X=X)
        
        return X,y
        
    def transform(self,X):
        
        # Breaks links
        X = X.copy()
    
        X = check_X(X=X)
        
        if not self.is_fitted:
            raise Exception("Object is not not fitted yet, please cal .fit(X,y)")
        
        if hasattr(X, "iloc"):
            X_targeted = X.iloc[:,self.mask_features]
        else:
            X_targeted = X[:,self.mask_features]
        
        return X_targeted
   


class LassoCV(LassoBase):
    """
    Compute Lasso path with cross-validation
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 method="lasso",
                 l1_ratio=0.5,
                 eps=0.001,
                 n_alphas=100,
                 alphas=None,
                 fit_intercept=True,
                 precompute='auto',
                 max_iter=1000,
                 tol=0.0001,
                 copy_X=True,
                 cv=None,
                 verbose=False,
                 n_jobs=None,
                 positive=False,
                 random_state=None,
                 selection='cyclic'):
        self.method = method
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.is_fitted = False

        # Sanity check
        bg.sanity_check.check_str(x=method,
                                  allowed=self.METHOD_ALLOWED,
                                  name="method")
    
        # Instantiate estimator
        if method=="lasso":
            self.estimator = ScikitLassoCV(
                eps=self.eps,
                n_alphas=self.n_alphas,
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
                precompute=self.precompute,
                max_iter=self.max_iter,
                tol=self.tol,
                copy_X=self.copy_X,
                cv=self.cv,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                positive=self.positive,
                random_state=self.random_state,
                selection=self.selection
                )
            
        elif method=="elasticnet":
            self.estimator = ScikitElasticNetCV(
                l1_ratio=self.l1_ratio,
                eps=self.eps,
                n_alphas=self.n_alphas,
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
                precompute=self.precompute,
                max_iter=self.max_iter,
                tol=self.tol,
                copy_X=self.copy_X,
                cv=self.cv,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                positive=self.positive,
                random_state=self.random_state,
                selection=self.selection
                )
    
    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------        
    METHOD_ALLOWED = ["lasso", "elasticnet"]
    
    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------
    def fit(self,X,y):
        X, y = super().preprocess(X=X,y=y)
        
        return super()._fit(X=X,y=y)
        
    def fit_transform(self,X,y):
        self.fit(X=X,y=y)
        
        return super().transform(X=X)
        
        
class LassoIC(LassoBase):
    """
    Compute Lasso path with information criterion
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 criterion='bic',
                 fit_intercept=True,
                 verbose=False,
                 precompute='auto',
                 max_iter=1000,
                 eps=2.220446049250313e-16,
                 copy_X=True,
                 positive=False,
                 noise_variance=None):        
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.precompute = precompute
        self.max_iter = max_iter
        self.eps = eps
        self.copy_X = copy_X
        self.positive = positive
        self.noise_variance = noise_variance
        self.is_fitted = False

        # Sanity check
        bg.sanity_check.check_str(x=criterion,
                                  allowed=self.CRITERION_ALLOWED,
                                  name="criterion")

        # Instantiate estimator
        self.estimator = LassoLarsIC(
            normalize=False, # Will deprecate in version 1.4.
            criterion=self.criterion,
            fit_intercept=self.fit_intercept,
            verbose=self.verbose,
            precompute=self.precompute,
            max_iter=self.max_iter,
            eps=self.eps,
            copy_X=self.copy_X,
            positive=self.positive,
            noise_variance=self.noise_variance
            )
        
    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    CRITERION_ALLOWED = ["bic", "aic"]
    
    # -------------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------------
    def _estimate_noise_variance(self, X, y):
        
        # X and y are already centered and we don't need to fit with an intercept
        variance_model = ScikitLassoCV(eps=0.001,
                                        n_alphas=10,
                                        alphas=None,
                                        fit_intercept=False,
                                        precompute='auto',
                                        max_iter=1000,
                                        tol=0.01,
                                        copy_X=True,
                                        cv=3,
                                        verbose=False,
                                        n_jobs=None,
                                        positive=False,
                                        random_state=None,
                                        selection='cyclic')

        # Get predictions
        y_pred = variance_model.fit(X, y).predict(X)

        # Compute effective degrees of freedom
        k = (variance_model.coef_ != 0).sum()
        
        degrees_of_freedom = X.shape[0] - k - self.fit_intercept
        
        noise_variance = np.sum((y - y_pred) ** 2) / degrees_of_freedom
                
        return noise_variance
           
    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------
    def fit(self,X,y):
        X, y = super().preprocess(X=X,y=y)
        
        if X.shape[0] <= (X.shape[1] + self.fit_intercept):
            if self.noise_variance is None:

                # Estimate noise variance
                noise_variance = self._estimate_noise_variance(X=X,y=y)
                
                # Overwrite                
                self.estimator.set_params(**{"noise_variance":noise_variance})
                        
        return super()._fit(X=X,y=y)
        
    def fit_transform(self,X,y):
        self.fit(X=X,y=y)
        
        return super().transform(X=X)
    
    

class LassoLambda(LassoBase):
    """
    Compute Lasso path with coordinate descent.
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 n_target,
                 fallback="force",
                 eps=0.001,
                 n_alphas=100,
                 alphas=None,
                 precompute='auto',
                 copy_X=True,
                 coef_init=None,
                 verbose=False,
                 return_n_iter=False,
                 positive=False,
                 tol=0.001,
                 max_iter=100000):
        self.n_target = n_target
        self.fallback = fallback
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.precompute = precompute
        self.copy_X = copy_X
        self.coef_init = coef_init
        self.verbose = verbose
        self.return_n_iter = return_n_iter
        self.positive = positive
        self.tol = tol
        self.max_iter = max_iter
        self.is_fitted = False
    
    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------        
    def fit(self,X,y):
                
        X, y = super().preprocess(X=X,y=y)
        
        # Run lasso model
        lambda_, coef_, _ = lasso_path(X=X, 
                                       y=y, 
                                       eps=self.eps,
                                       n_alphas=self.n_alphas,
                                       alphas=self.alphas,
                                       precompute=self.precompute,
                                       Xy=None,
                                       copy_X=self.copy_X,
                                       coef_init=self.coef_init,
                                       verbose=self.verbose,
                                       return_n_iter=self.return_n_iter,
                                       positive=self.positive,
                                       tol=self.tol,
                                       max_iter=self.max_iter)
                    
        # Count number of non-zero (absolute) coefficients for each lambda in path
        nonzero_bool = abs(coef_) !=0
        nonzero_coef = np.sum(nonzero_bool, axis=0)
                               
        # If any path along lambda gives equal (or greater) number of predictors than targeted, then use them. Otherwise take the least penalizing lambda.
        if np.any(nonzero_coef >= self.n_target):
                    
            # Pick first index with same (or greater) number of predictors as targeted
            lambda_idx = int(np.argwhere((nonzero_coef >= self.n_target) == True)[0])
            
            # Get boolean of features to include        
            mask_features = nonzero_bool[:,lambda_idx]
            
            if mask_features.sum()!=self.n_target:
                raise Exception(f"Number of selected features (={mask_features.sum()}) does not match target (={self.n_target})")
                            
        else:
            # There is no lambda that selects more predictors than targeted.
            # Hence, we choose the maximum number of predictors for any lambda (it cannot be higher than targeted)
            # In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
            lambda_idx = nonzero_coef.argmax()

            if self.fallback=="laissez_faire":
                # Get boolean of features to include        
                mask_features = nonzero_bool[:,lambda_idx]

            elif self.fallback=="force":
                
                # Number of features we need to include by force
                n_extra_features = self.n_target-nonzero_coef[lambda_idx]

                # Find features already selected
                features_already_selected = nonzero_bool[:,lambda_idx]                

                # Run OLS to get 'extra' coefficients
                features_importance = abs(np.array(OLS(fit_intercept=True).fit(X=X,y=y).test_stat_.iloc[1:]))
                
                # Subset to features not already selected
                features_left = features_importance[~features_already_selected]
                
                # Sort in place
                features_left = features_left[np.argsort(features_left)]

                # Find top features to look for based on importance                
                top_features = features_left[-n_extra_features:]
                
                mask_features_extra = np.where(np.in1d(features_importance, top_features))[0]
                
                # Pre-allocate
                features_extra_selected = np.array([False]*X.shape[1])
                
                # Overwrite
                features_extra_selected[mask_features_extra] = True
                
                # Combine
                mask_features = features_already_selected | features_extra_selected
                
        # At this point, we should have a vector of length p with True/False, indicating which features to use
        self.mask_features = mask_features
        
        # Update fitted status
        self.is_fitted = True
        
        return self

    def fit_transform(self,X,y):
        self.fit(X=X,y=y)
        
        return super().transform(X=X)

