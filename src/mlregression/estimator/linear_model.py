#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import bodyguard as bg
import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted
from ..utils.sanity_check import check_X_Y, check_X
import scipy as sp
#------------------------------------------------------------------------------
# OLS
#------------------------------------------------------------------------------
class OLS(BaseEstimator, RegressorMixin):
    """
    This class implements the classic OLS
    
    cov_type: 'nonrobust', ‘HC0’, ‘HC1’, ‘HC2’, ‘HC3’, or even cov_type='cluster', cov_kwds={'groups': mygroups}
    """
    
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 fit_intercept=True,
                 method="pinv",
                 cov_type="HC1",
                 cov_kwds=None,
                 use_t=True
                 ):        
        self.fit_intercept = fit_intercept
        self.method = method
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.use_t = use_t
        
    # -------------------------------------------------------------------------
    # Private functions
    # -------------------------------------------------------------------------
    def _extract_ols_info(self,fitted_estimator):
        
        # Extract headder from summary
        head = pd.read_html(fitted_estimator.summary2().as_html())[0]
        
        header_info = {}
        
        for col in [0, 2]:
            for row in head.index:
                header_info[head.iloc[row, col].replace(":", "")]=head.iloc[row, col+1]
        
        # Extract main body from summary
        main = pd.read_html(fitted_estimator.summary2().as_html())[1]
        
        # Fix rows
        main = main.T.set_index(0).T
        
        # Torn to numeric
        main = bg.tools.to_numeric(df=main)
        
        # Fix cols
        main = main.set_index(list(main)[0])
        
        # Set to strings
        main.index = [str(i) for i in main.index]
        
        main.index.names = ['']
    
        return main, header_info

    def _format_latex_table(self, model_fitted):
        
        # Generate raw output
        output = summary_col(results=[model_fitted],
               stars=True,
               float_format='%0.3f',
               info_dict={
                   "Observations":lambda x: "{0:d}".format(int(x.nobs)),
                   "Degrees of freedom":lambda x: "{0:d}".format(int(x.df_resid)),
                   "R2normal":lambda x: "{0:.3f}".format(x.rsquared),
                   "R2adj":lambda x: "{0:.3f}".format(x.rsquared_adj),
                   "F-stat":lambda x: "{:.3f}".format(x.fvalue),
                   "P(F-stat)":lambda x: "{:.3f}".format(x.f_pvalue),
                   "Log-Likelihood":lambda x: "{:.3f}".format(x.llf),
                   "AIC":lambda x: "{0:.3f}".format(x.aic),
                   "BIC":lambda x: "{0:.3f}".format(x.bic)
                   }
               )
            
        # Remove default R-squared
        for i,table in enumerate(output.tables):
            # Get mask of indices to include
            mask = np.array(["R-squared" not in idx for idx in table.index])
            
            # Generate new table without indices
            table_new = table.loc[mask]
            
            output.tables[i] = table_new
     

        # Render output
        output_rendered = output.as_latex()    
        
        # Replace table environment
        output_rendered = output_rendered.replace("\\begin{table}\n\\caption{}\n\\label{}\n\\begin{center}\n", "")
        output_rendered = output_rendered.replace("\n\\end{center}\n\\end{table}", "")
            
        # Change stats
        output_rendered = output_rendered.replace("R2normal", "$R^{2}$")
        output_rendered = output_rendered.replace("R2adj", "$R^{2}\\textrm{-adj}$")
        output_rendered = output_rendered.replace("F-stat", "$F\\textrm{-stat}$")
        
        # Replace hlines 
        output_rendered = bg.strings.nth_repl(s=output_rendered,
                                              old="hline",
                                              new="toprule",
                                              nth=1)

        output_rendered = bg.strings.nth_repl(s=output_rendered,
                                              old="hline",
                                              new="midrule",
                                              nth=1)

        output_rendered = bg.strings.nth_repl(s=output_rendered,
                                              old="hline",
                                              new="bottomrule",
                                              nth=1)        
        
        # Add midline after R-squared
        output_rendered = bg.strings.nth_repl(s=output_rendered,
                                              old="Observations",
                                              new="\\midrule\nObservations",
                                              nth=1)    

        return output_rendered


    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------        
    def fit(self,X,y):
        
        self.was_np_array = isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        
        # Check X and Y
        X, y = check_X_Y(X, y, convert_to_nparray=False)
        
        if self.fit_intercept:
            X = sm.add_constant(data=X,prepend=True, has_constant='skip')
        
        # Initialize estimator
        self.estimator = sm.OLS(endog=y,
                                exog=X,
                                missing='raise',
                                hasconst=None)
        
        # Fit
        self.estimator_fitted = self.estimator.fit(method=self.method,
                                                   cov_type=self.cov_type,
                                                   cov_kwds=self.cov_kwds,
                                                   use_t=self.use_t)
                                        
        # Predict in-sample
        self.y_hat_insample = self.estimator_fitted.predict(X)
        self.residuals_in_sample = y - self.y_hat_insample
        
        if self.was_np_array:
            self.y_hat_insample = np.array(self.y_hat_insample)
            self.residuals_in_sample = np.array(self.residuals_in_sample)
        
        # Extract results
        self.result_, self.info_ = self._extract_ols_info(fitted_estimator=self.estimator_fitted)
        
        # Save attributes
        self.coef_ = self.result_.iloc[:,0]
        self.se_ = self.result_.iloc[:,1]    
        self.test_stat_ = self.result_.iloc[:,2]
        self.p_value_ = self.result_.iloc[:,3]
        
        self.sse_sum_of_squares_error_ = self.estimator_fitted.ssr
        self.ssr_sum_of_squares_regression_ = self.estimator_fitted.ess
        self.sst_sum_of_squares_total_ = self.sse_sum_of_squares_error_+self.ssr_sum_of_squares_regression_
        
        # Collect
        self.regression_table_ = pd.concat(objs=[self.coef_,self.se_,self.test_stat_,self.p_value_],
                                           axis=1,
                                           ignore_index=False)
        
        self.regression_table_.index.rename(name="Variable", inplace=True)
        
        for k,v in self.info_.items():
            
            # Prettify string
            attr_name = bg.strings.make_string_compatible(string=k)
            
            if not attr_name.endswith("_"):
                attr_name = attr_name+"_"
            
            # Set attribute
            setattr(self,attr_name,v)
            
            
            
            
        
        return self
    
    def to_latex(self):
        
        # Render output
        output_rendered = self._format_latex_table(model_fitted=self.estimator_fitted)
        
        return output_rendered
        
    
    def summary(self, show_summary=True):
        
        self.summary_ = self.estimator_fitted.summary()
        
        if show_summary:
            print(self.summary_)
        
    def predict(self,X):
        
        # Check X
        X = check_X(X, convert_to_nparray=False)
        
        if self.fit_intercept:
            X = sm.add_constant(data=X,prepend=True, has_constant='skip')
        
        y_hat = self.estimator_fitted.predict(exog=X)
        
        return y_hat
       
    
#------------------------------------------------------------------------------
# Constrained OLS
#------------------------------------------------------------------------------
class ConstrainedLS(BaseEstimator, RegressorMixin):
    """
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 fit_intercept=True,
                 constraints=None,
                 loss_fnc="sum_squares",
                 max_iters=10000,
                 solver=cp.SCS,
                 verbose=False,
                  ):
        self.fit_intercept = fit_intercept
        self.constraints = constraints
        self.loss_fnc = loss_fnc
        self.max_iters = max_iters
        self.solver = solver
        self.verbose = bool(verbose) #SCS/ECOS do not handle integer verbose
        
        # Initiate other variables
        self.X_name = None
        self.y_name = None
        
        # ---------------------------------------------------------------------
        # Sanity checks
        # ---------------------------------------------------------------------
        bg.sanity_check.check_str(x=self.loss_fnc,
                                  allowed=self.ALLOWED_LOSS_FNC,
                                  name="loss_fnc")    
        
    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------        
    ALLOWED_DATA_TYPE = (np.ndarray)
    ALLOWED_LOSS_FNC = ["nuc", "fro", "inf", "spectral", "sum_squares"]
    ALLOWED_DIMENSION_REDUCTION_METHODS = ["pca", "average"]
    CONSTRAINT_KEYS = ["sum",
                       "sum_lower_bound", "sum_upper_bound",
                       "lower_bound", "upper_bound"]

    # --------------------
    # Private
    # --------------------
    def _score(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def _prepare_constraints(self, constraints, var):
        
        constraints_compatible = []
        if constraints is not None:        
            for k,v in constraints.items():
                if k=="sum":
                    constraints_compatible += [cp.sum(var) == v]
                elif k=="sum_lower_bound":
                    constraints_compatible += [cp.sum(var) >= v]
                elif k=="sum_upper_bound":
                    constraints_compatible += [cp.sum(var) <= v]                                                            
                elif k=="lower_bound":
                    constraints_compatible += [var >= v]
                elif k=="upper_bound":
                    constraints_compatible += [var <= v]                    
                else:
                    raise Exception(f"Constraint keys is restricted to one of {self.CONSTRAINT_KEYS}, but one key is {k}")
    
        return constraints_compatible

    def _general_loss(self,x):
        """ Generate general loss without regularization """
        
        if self.loss_fnc in ["nuc", "fro", "inf"]:
            loss = cp.norm(x=x, p=self.loss_fnc)
            
        elif self.loss_fnc=="spectral":
            loss = cp.norm(x=x, p=2)
            
        elif self.loss_fnc=="sum_squares":
            loss = cp.sum_squares(x)
            
        return loss


    def _prepare_loss(self,y,X,beta):
        
        """ Generate loss = main_loss"""
        # Estimate input to norm
        loss_input = y - X @ beta
        
        loss = self._general_loss(x=loss_input)
                   
        return loss

    def _solve(self, loss, constraints):
          
        # Set up objective function
        Q = cp.Minimize(loss)

        # Set up problem
        problem = cp.Problem(objective=Q, constraints=constraints)

        # Solve
        problem.solve(verbose=self.verbose,
                      solver=self.solver,
                      ignore_dpp=True,
                      max_iters=self.max_iters)

    def _extract_value(self, x):
        if isinstance(x, cp.expressions.variable.Variable):
            x = x.value

        if sp.sparse.issparse(x):
            x = x.toarray()
            
        return x
    # --------------------
    # Public functions
    # --------------------
    def fit(self, X, y):

        if self.fit_intercept:
            X = sm.add_constant(data=X,prepend=True, has_constant='skip')
        
        if isinstance(y, pd.Series):
            self.y_name = y.name

        if isinstance(X, pd.DataFrame):
            self.X_name = X.columns.tolist()

        # Check X and Y, incl. break link, check dimensions and transform to np.array
        X, y = check_X_Y(X, y)
        
        # Set up decision variables, i.e. beta coefficients
        beta = cp.Variable(shape=(X.shape[1],), name="beta")

        # Constraints
        constraints = self._prepare_constraints(constraints=self.constraints,
                                                var=beta)

        loss = self._prepare_loss(y=y,X=X,beta=beta)

        self._solve(loss=loss,
                    constraints=constraints)

        self.beta_ = self._extract_value(x=beta)
        
        if self.X_name is not None:
            self.beta_ = pd.Series(self.beta_, index=self.X_name)
        
        # Fitted valued
        self.y_fitted_ = X @ self.beta_

        # Return mean squared error
        self.best_score_ = self._score(y_true=y, y_pred=self.y_fitted_)
        
        return self
        
        
    def predict(self, X):
        
        # Check is fit had been called
        check_is_fitted(self)
        
        if self.fit_intercept:
            X = sm.add_constant(data=X,prepend=True, has_constant='skip')
        
        # Check X
        X = check_X(X)
        
        y_hat = X @ self.beta_
                
        if self.y_name is not None:
            y_hat = pd.Series(y_hat, name=self.y_name)

        return y_hat

    def score(self, X, y, sample_weight=None):
        # Predict
        y_pred = self.predict(X)
        
        return self._score(y_true=y, y_pred=y_pred)
            